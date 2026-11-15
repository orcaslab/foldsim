# Physical domains as placement units (foldsim --physical-domains --domain-pblocks): each SLR group's root domains
# (fs_dom_<class> instances right under foldsim_slr<g>, in the class tree's order) get side-by-side column stripes of
# their group's SLR, each stripe the SLR's full height (every domain reaches both SLR boundaries) and as wide as the
# domain's share of the SLR's sites of every kind it uses (LUT, block RAM, ultra RAM, DSP), measured on the open
# synthesized design; a domain with child domains is striped again inside its own stripe (nested pblocks, as the
# domains nest). Soft pblocks: the placer may spill a cell
# rather than fail. Sourced after floorplan.tcl (the groups' USER_SLR_ASSIGNMENT) by ooc.tcl / pre_opt_hook.tcl when
# FOLDSIM_DOMAIN_PBLOCKS names the driver's foldsim_domain_pblocks.tcl (the data: fs_domain_roots).
#
# fs_domain_roots: {SLR {root cell name suffix ...}} pairs, e.g. {SLR1 {foldsim_slr1/fs_dom_X foldsim_slr1/fs_dom_Y}}.

proc fs_dom_tile_x {site} {
  set tl [lindex [get_tiles -quiet -of_objects $site] 0]
  if {[regexp {_X(\d+)Y\d+$} $tl -> x]} { return $x }
  return -1
}

# Each kind's sites in the SLR by tile column: dict kind -> dict x -> count; and per kind its sites by x for ranges.
proc fs_dom_columns {slr} {
  set cols [dict create]
  foreach {kind pat} {lut SLICE* bram RAMB36* uram URAM288* dsp DSP*} {
    set by [dict create]
    foreach s [get_sites -quiet -of_objects [get_slrs $slr] -filter "SITE_TYPE =~ $pat"] {
      set x [fs_dom_tile_x $s]
      dict lappend by $x $s
    }
    dict set cols $kind $by
  }
  return $cols
}

proc fs_dom_demand {cell} {
  set d [dict create]
  dict set d lut [llength [get_cells -quiet -hierarchical -filter "NAME =~ $cell/* && PRIMITIVE_SUBGROUP == LUT"]]
  dict set d bram [llength [get_cells -quiet -hierarchical -filter "NAME =~ $cell/* && PRIMITIVE_GROUP == BLOCKRAM && REF_NAME =~ RAMB36*"]]
  dict set d uram [llength [get_cells -quiet -hierarchical -filter "NAME =~ $cell/* && REF_NAME =~ URAM288*"]]
  dict set d dsp [llength [get_cells -quiet -hierarchical -filter "NAME =~ $cell/* && PRIMITIVE_GROUP == ARITHMETIC"]]
  return $d
}

# Stripes for `cells` (sibling domains) over the tile columns xs (sorted) of the SLR, left to right, each as wide as
# its share of every kind it uses; `parent` (a demand dict, or "") is the enclosing domain's: its children take their
# share of the parent's stripe and the parent's own logic the rest (at the right). A domain with two or more child
# domains (fs_dom_* instances) is striped again inside its own stripe: the pblocks nest as the domains do.
proc fs_dom_stripes {slr cols cells xs parent} {
  global fs_dom_k
  set dem [dict create]; set tot [dict create lut 0 bram 0 uram 0 dsp 0]
  foreach r $cells {
    set d [fs_dom_demand $r]
    dict set dem $r $d
    foreach k {lut bram uram dsp} { dict incr tot $k [dict get $d $k] }
  }
  # the share base: the parent's demand (its own logic keeps the rest of its stripe), else the siblings' total
  set base [expr {$parent eq "" ? $tot : $parent}]
  set cap [dict create lut 0 bram 0 uram 0 dsp 0]
  foreach x $xs { foreach k {lut bram uram dsp} {
    if {[dict exists [dict get $cols $k] $x]} { dict incr cap $k [llength [dict get [dict get $cols $k] $x]] } } }
  set i 0; set at 0; set nx [llength $xs]; set out {}
  foreach r $cells {
    incr i
    if {$at >= $nx} { puts "CRITICAL WARNING: foldsim domain pblocks: no column left for $r"; break }
    set have [dict create lut 0 bram 0 uram 0 dsp 0]
    set from $at
    if {$i == [llength $cells] && $parent eq ""} { set at $nx } else {
      while {$at < $nx} {
        set x [lindex $xs $at]
        foreach k {lut bram uram dsp} {
          if {[dict exists [dict get $cols $k] $x]} { dict incr have $k [llength [dict get [dict get $cols $k] $x]] }
        }
        incr at
        set ok 1
        foreach k {lut bram uram dsp} {
          set t [dict get $base $k]
          if {$t == 0} continue
          set want [expr {double([dict get [dict get $dem $r] $k]) / $t * [dict get $cap $k]}]
          if {[dict get $have $k] < $want} { set ok 0 }
        }
        if {$ok} break
      }
    }
    lappend out [list $r [lrange $xs $from [expr {$at - 1}]] [dict get $dem $r]]
  }
  foreach s $out {
    lassign $s r rxs d
    set x0 [lindex $rxs 0]; set x1 [lindex $rxs end]
    set pb [create_pblock fs_dom_pb_[incr fs_dom_k]]
    foreach kind {lut bram uram dsp} {
      set sites {}
      dict for {x l} [dict get $cols $kind] { if {$x >= $x0 && $x <= $x1} { lappend sites {*}$l } }
      if {[llength $sites] == 0} continue
      foreach p [lsort -unique [lmap st $sites {regsub {_X\d+Y\d+$} $st ""}]] {
        set nx {}; set ny {}
        foreach st $sites { if {[regexp "^${p}_X(\\d+)Y(\\d+)\$" $st -> sx sy]} { lappend nx $sx; lappend ny $sy } }
        set nx [lsort -integer $nx]; set ny [lsort -integer $ny]
        resize_pblock $pb -add "${p}_X[lindex $nx 0]Y[lindex $ny 0]:${p}_X[lindex $nx end]Y[lindex $ny end]"
      }
    }
    set_property IS_SOFT TRUE $pb
    add_cells_to_pblock $pb [get_cells $r]
    puts "INFO: foldsim domain pblock $pb: $r tile columns $x0..$x1 ($slr), demand $d"
    set kids [get_cells -quiet $r/fs_dom_*]
    if {[llength $kids] >= 2 && [llength $rxs] >= 2} { fs_dom_stripes $slr $cols $kids $rxs $d }
  }
}

set fs_dom_k 0
if {[info exists ::env(FOLDSIM_DOMAIN_PBLOCKS)] && $::env(FOLDSIM_DOMAIN_PBLOCKS) ne ""} {
  source $::env(FOLDSIM_DOMAIN_PBLOCKS)
  foreach {slr roots} $fs_domain_roots {
    set cells {}
    foreach r $roots {
      set c [get_cells -quiet $r]
      if {[llength $c] != 1} { set c [get_cells -quiet -hierarchical -filter "NAME =~ */$r"] }
      if {[llength $c] == 1} { lappend cells $c } else { puts "CRITICAL WARNING: foldsim domain $r: [llength $c] cells" }
    }
    set cols [fs_dom_columns $slr]
    set xs {}
    foreach kind {lut bram uram dsp} { foreach x [dict keys [dict get $cols $kind]] { if {$x >= 0} { lappend xs $x } } }
    set xs [lsort -integer -unique $xs]
    # a single root: striped inside (its children), else the roots side by side
    if {[llength $cells] == 1} {
      set kids [get_cells -quiet [lindex $cells 0]/fs_dom_*]
      if {[llength $kids] >= 2} { fs_dom_stripes $slr $cols $kids $xs [fs_dom_demand [lindex $cells 0]] }
    } elseif {[llength $cells] > 1} { fs_dom_stripes $slr $cols $cells $xs "" }
  }
}
