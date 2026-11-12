# Applies foldsim's --slr-groups floorplan, $::env(FOLDSIM_FLOORPLAN), to the
# open synthesized design: each threaded copy kept whole in one SLR, the SLR
# left to the placer. A copy split over an SLR boundary pays a crossing on
# paths wholly inside it. Sourced before opt_design by ooc.tcl and by the
# image's implementation (pre_opt_hook.tcl); the Makefile sets the variable
# when FOLDSIM_FLAGS has --slr-groups. The assignments land in floorplan.txt.

if {[info exists ::env(FOLDSIM_FLOORPLAN)] && $::env(FOLDSIM_FLOORPLAN) ne ""} {
    set fs_floorplan $::env(FOLDSIM_FLOORPLAN)
    if {![file exists $fs_floorplan]} {
        error "foldsim floorplan not found: $fs_floorplan"
    }
    set fp [open $fs_floorplan r]
    # The copies' groups (foldsim_slr<n>); the virtual-memory controllers
    # placed with them (--vmem-engines) are cells of the kernel, absent from
    # an out-of-context run of the simulator alone.
    set fs_groups [regexp -all {hierarchical foldsim_slr} [read $fp]]
    close $fp
    source $fs_floorplan
    # (A floorplan names either placer-chosen groups, fs_slr<g>, or the region's SLRs by name, SLR<n>, with --slr-region.)
    set fs_cells [get_cells -quiet -hierarchical -filter {USER_SLR_ASSIGNMENT =~ "fs_*" || USER_SLR_ASSIGNMENT =~ "SLR*"}]
    set fp [open floorplan.txt w]
    foreach c $fs_cells { puts $fp "$c [get_property USER_SLR_ASSIGNMENT $c]" }
    close $fp
    # A floorplan that matches nothing would leave the copies to area-only
    # placement without a trace in the reports.
    if {[llength $fs_cells] == 0} {
        error "foldsim floorplan $fs_floorplan matched no cell"
    }
    if {[llength $fs_cells] < $fs_groups} {
        puts "CRITICAL WARNING: foldsim floorplan: [llength $fs_cells] of $fs_groups copies found"
    }
    puts "INFO: foldsim floorplan: [llength $fs_cells] threaded copies in SLR groups ($fs_floorplan)"
}
