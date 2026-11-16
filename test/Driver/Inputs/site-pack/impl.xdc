create_pblock pblock_slash
resize_pblock [get_pblocks pblock_slash] -add {URAM288_X0Y2:URAM288_X0Y23 URAM288_X2Y0:URAM288_X2Y35}
resize_pblock [get_pblocks pblock_slash] -add {RAMB36_X0Y0:RAMB36_X0Y35}
resize_pblock [get_pblocks pblock_slash] -add {CLOCKREGION_X1Y3:CLOCKREGION_X1Y3}
set_property PROHIBIT true [get_sites {URAM288_X1Y30:URAM288_X1Y31}]
