# Vivado stubs for flow/v80/domain_pblocks.tcl's stripe procedure: a three-SLR-group domain tree like the 16-SM
# fold's SLR0 group (one root past the least stripe, VX_core, with small sibling roots), its demands in LUTs, and an
# SLR of 100 rows (x 0..99) of 10 LUT sites each. Prints each pblock made: its cells and its row range.
set tree {
  g2/fs_dom_kmub {}
  g2/fs_dom_l2fs_0 {}
  g2/fs_dom_core {g2/fs_dom_core/fs_dom_cs g2/fs_dom_core/fs_dom_lms}
  g2/fs_dom_core/fs_dom_cs {g2/fs_dom_core/fs_dom_cs/fs_dom_alu g2/fs_dom_core/fs_dom_cs/fs_dom_sfu}
  g2/fs_dom_core/fs_dom_lms {}
  g2/fs_dom_core/fs_dom_cs/fs_dom_alu {}
  g2/fs_dom_core/fs_dom_cs/fs_dom_sfu {}
}
set luts {g2/fs_dom_kmub 2111 g2/fs_dom_l2fs_0 3824 g2/fs_dom_core 179878 g2/fs_dom_core/fs_dom_cs 84356
          g2/fs_dom_core/fs_dom_lms 1728 g2/fs_dom_core/fs_dom_cs/fs_dom_alu 32716 g2/fs_dom_core/fs_dom_cs/fs_dom_sfu 20510}
proc get_cells {args} {
  global tree
  set a [lindex $args end]
  if {[string match */fs_dom_* $a] && [string index $a end] eq "*"} { return [dict get $tree [string range $a 0 end-9]] }
  return $a
}
proc get_pblocks {args} { return "" }
proc get_property {args} { return "" }
proc create_pblock {n} { return $n }
proc resize_pblock {pb args} { global range; set range($pb) [lindex $args end] }
proc set_property {args} {}
proc add_cells_to_pblock {pb c} { global made range; lappend made "$pb $c" }
source [lindex $argv 0]
proc fs_dom_demand {cell} { global luts; return [dict create lut [dict get $luts $cell] lutram 0 bram 0 uram 0 dsp 0] }
proc fs_dom_enclosing {cell} { return "" }
set cols [dict create lut {} lutram {} bram {} uram {} dsp {}]
set xs {}
for {set x 0} {$x < 100} {incr x} {
  lappend xs $x
  set l {}; for {set y 0} {$y < 10} {incr y} { lappend l SLICE_X${y}Y${x} }
  dict set cols lut [dict merge [dict get $cols lut] [dict create $x $l]]
}
set made {}
fs_dom_stripes SLR0 $cols {g2/fs_dom_kmub g2/fs_dom_core g2/fs_dom_l2fs_0} $xs ""
foreach m $made { puts "PBLOCK $m" }
puts "PBLOCKS [llength $made]"
