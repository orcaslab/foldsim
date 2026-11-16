# Physical domains as placement units (foldsim --physical-domains --domain-pblocks): each SLR group's root domains
# (fs_dom_<class> instances right under foldsim_slr<g>, in the class tree's order) get side-by-side column stripes of
# their group's SLR (fs_domain_axis x), or stacked row bands ordered by their crossings' direction (fs_domain_axis y:
# a domain whose links cross to the SLR above sits at the top, so they land at the boundary), each as wide as the
# domain's share of the SLR's sites of every kind it uses (LUT, block RAM, ultra RAM, DSP), measured on the open
# synthesized design; a domain with child domains is striped again inside its own stripe (nested pblocks, as the
# domains nest). Soft pblocks: the placer may spill a cell
# rather than fail. The driver appends this file, after its data (fs_domain_roots), to the gen's foldsim_floorplan.tcl,
# which the flows source after the groups' SLR assignment (ooc.tcl, pre_opt_hook.tcl): no flow change. Alone it
# reads the data from FOLDSIM_DOMAIN_PBLOCKS (the driver's foldsim_domain_pblocks.tcl).
#
# fs_domain_roots: {SLR {root cell name suffix ...}} pairs, e.g. {SLR1 {foldsim_slr1/fs_dom_X foldsim_slr1/fs_dom_Y}}.


# Each kind's sites in the SLR by tile column: dict kind -> dict x -> count; and per kind its sites by x for ranges.
proc fs_dom_enclosing {cell} {
  # the pblock the cell or its nearest ancestor is in (a reconfigurable module's region, a per-SLR pblock), or ""
  set c $cell
  while {$c ne ""} {
    set pb [get_pblocks -quiet -of_objects $c]
    if {[llength $pb] == 1} { return $pb }
    set p [get_property PARENT $c]
    if {$p eq "" || $p eq [get_property NAME $c]} { break }
    set c [get_cells -quiet $p]
  }
  return ""
}

# The SLR's sites of each kind (inside `parent`'s, when the domains sit in a pblock) by coordinate on `axis` (the
# sites' RPM grid, read in bulk: one coordinate system for every site kind).
proc fs_dom_columns {slr axis parent} {
  set prop [expr {$axis eq "y" ? "RPM_Y" : "RPM_X"}]
  set cols [dict create]
  foreach {kind pat} {lut SLICE* lutram SLICEM bram RAMB36* uram URAM288* dsp DSP*} {
    set by [dict create]
    set ss [get_sites -quiet -of_objects [get_slrs $slr] -filter "SITE_TYPE =~ $pat"]
    if {[llength $ss]} {
      # (inside a pblock: its sites of the kind within the SLR's rows -- an SLR is a band of rows)
      set sites $ss
      if {$parent ne ""} {
        set ys [lsort -integer [get_property RPM_Y $ss]]
        set lo [lindex $ys 0]; set hi [lindex $ys end]
        set sites {}
        set ps [get_sites -quiet -of_objects $parent -filter "SITE_TYPE =~ $pat"]
        if {[llength $ps]} {
          foreach s $ps y [get_property RPM_Y $ps] { if {$y >= $lo && $y <= $hi} { lappend sites $s } }
        }
      }
      if {[llength $sites]} {
        foreach s $sites x [get_property $prop $sites] { dict lappend by $x $s }
      }
    }
    dict set cols $kind $by
  }
  return $cols
}

# Pblock ranges covering exactly `sites`: per site kind and column, each run of consecutive rows one rectangle.
proc fs_dom_ranges {sites} {
  set runs [dict create]
  foreach st $sites {
    if {[regexp {^(\w+?)_X(\d+)Y(\d+)$} $st -> p sx sy]} { dict lappend runs "$p $sx" $sy }
  }
  set out {}
  dict for {k ys} $runs {
    lassign $k p sx
    set ys [lsort -integer -unique $ys]
    set a [lindex $ys 0]; set b $a
    foreach y [lrange $ys 1 end] {
      if {$y == $b + 1} { set b $y; continue }
      lappend out "${p}_X${sx}Y${a}:${p}_X${sx}Y${b}"; set a $y; set b $y
    }
    lappend out "${p}_X${sx}Y${a}:${p}_X${sx}Y${b}"
  }
  return $out
}

proc fs_dom_demand {cell} {
  # One utilization report of the cell (a few seconds even on a full device); its LUT, RAMB36-equivalent, URAM and DSP
  # rows. A row the report lacks counts the leaf cells instead.
  set d [dict create lut 0 lutram 0 bram 0 uram 0 dsp 0]
  set rpt [report_utilization -quiet -cells [get_cells $cell] -return_string]
  set got 0
  foreach {k re} {lut {\|\s*CLB LUTs\*?\s*\|\s*(\d+)} lutram {\|\s*LUT as (?:Distributed RAM|Memory)\s*\|\s*(\d+)} bram {\|\s*Block RAM Tile\s*\|\s*([\d.]+)}
                  uram {\|\s*URAM\s*\|\s*(\d+)} dsp {\|\s*DSP[\w ]*\|\s*(\d+)}} {
    if {[regexp $re $rpt -> v]} { dict set d $k [expr {int(ceil($v))}]; incr got }
  }
  if {$got == 0} {
    dict set d lut [llength [get_cells -quiet -hierarchical -filter "NAME =~ $cell/* && PRIMITIVE_SUBGROUP == LUT"]]
    dict set d bram [llength [get_cells -quiet -hierarchical -filter "NAME =~ $cell/* && REF_NAME =~ RAMB36*"]]
    dict set d uram [llength [get_cells -quiet -hierarchical -filter "NAME =~ $cell/* && REF_NAME =~ URAM288*"]]
    dict set d dsp [llength [get_cells -quiet -hierarchical -filter "NAME =~ $cell/* && PRIMITIVE_GROUP == ARITHMETIC"]]
  }
  return $d
}

# Stripes for `cells` (sibling domains) over the tile columns xs (sorted) of the SLR, left to right, each as wide as
# its share of every kind it uses; `parent` (a demand dict, or "") is the enclosing domain's: its children take their
# share of the parent's stripe and the parent's own logic the rest (at the right). A domain with two or more child
# domains (fs_dom_* instances) is striped again inside its own stripe: the pblocks nest as the domains do.
proc fs_dom_aff {cell} {
  global fs_domain_affinity
  set a 0
  if {[info exists fs_domain_affinity]} {
    foreach {path v} $fs_domain_affinity { if {[string match "*$path" $cell]} { set a $v } }
  }
  return $a
}

proc fs_dom_stripes {slr cols cells xs parent {pcell ""}} {
  global fs_dom_k fs_dom_min_lut
  set dem [dict create]; set tot [dict create lut 0 lutram 0 bram 0 uram 0 dsp 0]
  set big {}
  foreach r $cells {
    set d [fs_dom_demand $r]
    # (a domain below the least stripe, FOLDSIM_DOMAIN_MIN_LUT LUTs, stays in its parent's area, a root in its SLR: a stripe a column
    # wide and the SLR high is the wrong shape for it)
    if {[dict get $d lut] < $fs_dom_min_lut} continue
    lappend big $r
    dict set dem $r $d
    foreach k {lut lutram bram uram dsp} { dict incr tot $k [dict get $d $k] }
  }
  set cells $big
  # Roots of an SLR of which one alone is past the least stripe: that root is the SLR's single root, striped inside
  # (its child domains), as when it is the only root (the small roots stay in the SLR's area).
  if {$parent eq "" && [llength $cells] == 1} {
    set r [lindex $cells 0]
    set kids [get_cells -quiet $r/fs_dom_*]
    if {[llength $kids] >= 1} { fs_dom_stripes $slr $cols $kids $xs [dict get $dem $r] $r }
    return
  }
  if {[llength $cells] < ($parent eq "" ? 2 : 1)} return
  # Bands (fs_domain_axis y): bottom to top by each domain's affinity (its link wires to the SLRs above minus those to
  # the SLRs below, fs_domain_affinity, by instance path suffix): a domain lands next to the boundary its crossings use.
  global fs_domain_axis fs_domain_affinity
  if {[info exists fs_domain_axis] && $fs_domain_axis eq "y" && [info exists fs_domain_affinity]} {
    set keyed {}; set sum 0
    foreach r $cells { set a [fs_dom_aff $r]; incr sum $a; lappend keyed [list $a $r] }
    set cells [lmap e [lsort -integer -index 0 $keyed] {lindex $e 1}]
    # The parent's own logic keeps the rest of its band, at the end its own crossings point to (its affinity less its
    # banded children's): toward the SLR below, the children are stacked from the top down instead.
    if {$parent ne "" && $pcell ne "" && [fs_dom_aff $pcell] - $sum < 0} {
      set cells [lreverse $cells]
      set xs [lreverse $xs]
    }
  }
  # the share base: the parent's demand (its own logic keeps the rest of its stripe), else the siblings' total
  set base [expr {$parent eq "" ? $tot : $parent}]
  set cap [dict create lut 0 lutram 0 bram 0 uram 0 dsp 0]
  foreach x $xs { foreach k {lut lutram bram uram dsp} {
    if {[dict exists [dict get $cols $k] $x]} { dict incr cap $k [llength [dict get [dict get $cols $k] $x]] } } }
  # Each domain's need: its largest share of the base over the kinds it uses (a URAM-heavy domain needs as many rows
  # as its URAM share, whatever its LUTs); the parent's own logic needs its own largest share. When the needs add up
  # past the band (sites of one kind cluster in columns: a row band holds every kind in proportion), all shrink alike
  # (soft pblocks: the placer may spill).
  set need [dict create]; set sum 0.0
  foreach r $cells {
    set n 0.0
    foreach k {lut lutram} {
      set t [dict get $base $k]
      if {$t > 0 && [dict get $cap $k] > 0} { set n [expr {max($n, double([dict get [dict get $dem $r] $k]) / $t)}] }
    }
    dict set need $r $n; set sum [expr {$sum + $n}]
  }
  if {$parent ne ""} {
    set n 0.0
    foreach k {lut lutram} {
      set t [dict get $base $k]; set own $t
      foreach r $cells { set own [expr {$own - [dict get [dict get $dem $r] $k]}] }
      if {$t > 0 && [dict get $cap $k] > 0} { set n [expr {max($n, double(max(0, $own)) / $t)}] }
    }
    set sum [expr {$sum + $n}]
  }
  set scale [expr {$sum > 1.0 ? 1.0 / $sum : 1.0}]
  set i 0; set at 0; set nx [llength $xs]; set out {}
  foreach r $cells {
    incr i
    if {$at >= $nx} { puts "CRITICAL WARNING: foldsim domain pblocks: no row/column left for $r"; break }
    set have 0
    set from $at
    set want [expr {[dict get $need $r] * $scale * [dict get $cap lut]}]
    if {$i == [llength $cells] && $parent eq ""} { set at $nx } else {
      while {$at < $nx} {
        set x [lindex $xs $at]
        if {[dict exists [dict get $cols lut] $x]} { incr have [llength [dict get [dict get $cols lut] $x]] }
        incr at
        if {$have >= $want} break
      }
    }
    lappend out [list $r [lrange $xs $from [expr {$at - 1}]] [dict get $dem $r]]
  }
  foreach s $out {
    lassign $s r rxs d
    set x0 [tcl::mathfunc::min {*}$rxs]; set x1 [tcl::mathfunc::max {*}$rxs]
    set pb [create_pblock fs_dom_pb_[incr fs_dom_k]]
    # (logic sites of the band; block RAM, ultra RAM and DSP sites of the whole region: a row band holds them only in
    # proportion, and a RAM-heavy domain next to logic-heavy ones would be short of them -- a pblock past its RAM
    # capacity stops placement even when soft, Place 30-2958. The RAMs follow their logic.)
    set sites {}
    dict for {x l} [dict get $cols lut] { if {$x >= $x0 && $x <= $x1} { lappend sites {*}$l } }
    foreach kind {bram uram dsp} { dict for {x l} [dict get $cols $kind] { lappend sites {*}$l } }
    resize_pblock $pb -add [fs_dom_ranges $sites]
    set encl [fs_dom_enclosing $r]
    if {$encl ne ""} { set_property PARENT [get_property NAME $encl] $pb }
    set_property IS_SOFT TRUE $pb
    add_cells_to_pblock $pb [get_cells $r]
    # (the virtual-memory controllers of the domain's class, fs_domain_adapters: the kernel's cells, with their logic)
    global fs_domain_adapters
    if {[info exists fs_domain_adapters] && [regexp {fs_dom_(.*)$} $r -> cls] && [dict exists $fs_domain_adapters $cls]} {
      foreach re [dict get $fs_domain_adapters $cls] {
        set ac [get_cells -quiet -hierarchical -regexp $re]
        if {[llength $ac]} { add_cells_to_pblock $pb $ac; puts "INFO: foldsim domain pblock $pb: [llength $ac] adapter cells of $cls ($re)" }
      }
    }
    puts "INFO: foldsim domain pblock $pb: $r tile columns $x0..$x1 ($slr), demand $d"
    set kids [get_cells -quiet $r/fs_dom_*]
    if {[llength $kids] >= 1 && [llength $rxs] >= 2} { fs_dom_stripes $slr $cols $kids [lsort -integer $rxs] $d $r }
  }
}

set fs_dom_k 0
set fs_dom_min_lut [expr {[info exists ::env(FOLDSIM_DOMAIN_MIN_LUT)] ? $::env(FOLDSIM_DOMAIN_MIN_LUT) : 20000}]
if {[info exists ::env(FOLDSIM_DOMAIN_PBLOCKS)] && $::env(FOLDSIM_DOMAIN_PBLOCKS) ne "" && ![info exists fs_domain_roots]} {
  source $::env(FOLDSIM_DOMAIN_PBLOCKS)
}
if {[info exists fs_domain_roots]} {
  foreach {slr roots} $fs_domain_roots {
    set cells {}
    foreach r $roots {
      set c [get_cells -quiet $r]
      if {[llength $c] != 1} { set c [get_cells -quiet -hierarchical -filter "NAME =~ */$r"] }
      if {[llength $c] == 1} { lappend cells $c } else { puts "CRITICAL WARNING: foldsim domain $r: [llength $c] cells" }
    }
    puts "INFO: foldsim domain pblocks: $slr: [llength $cells] root domains, sites by [expr {[info exists fs_domain_axis] ? $fs_domain_axis : "x"}] ([clock format [clock seconds] -format %T])"
    set cols [fs_dom_columns $slr [expr {[info exists fs_domain_axis] ? $fs_domain_axis : "x"}] \
                  [expr {[llength $cells] ? [fs_dom_enclosing [lindex $cells 0]] : ""}]]
    set xs {}
    foreach x [dict keys [dict get $cols lut]] { if {$x >= 0} { lappend xs $x } }
    set xs [lsort -integer -unique $xs]
    puts "INFO: foldsim domain pblocks: $slr: sites read ([clock format [clock seconds] -format %T])"
    # a single root: striped inside (its children), else the roots side by side
    if {[llength $cells] == 1} {
      set kids [get_cells -quiet [lindex $cells 0]/fs_dom_*]
      if {[llength $kids] >= 1} { fs_dom_stripes $slr $cols $kids $xs [fs_dom_demand [lindex $cells 0]] [lindex $cells 0] }
    } elseif {[llength $cells] > 1} { fs_dom_stripes $slr $cols $cells $xs "" }
  }
}
