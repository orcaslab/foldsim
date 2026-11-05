SECTIONS {
  . = 0x0;
  .text : { *(.text.start) *(.text*) *(.rodata*) *(.srodata*) }
  .data : { *(.data*) *(.sdata*) *(.bss*) *(.sbss*) }
  ASSERT(. <= 0x800, "program over 2 KiB")
}
