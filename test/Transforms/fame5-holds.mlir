// RUN: foldsim-opt %s --foldsim-fame5="module=Hold threads=4" | FileCheck %s

// A register that holds its value unless enabled is written only when
// enabled, with the new value, rather than written back every cycle.
// CHECK-LABEL: hw.module @Hold_threaded_Hold(
// CHECK: seq.firmem.write_port %r[%threadIdx] = %d_sink_bits, clock %clk_buffer.O enable %en_sink_bits

// One that holds unless reset or enabled is written when either is.
// CHECK: [[EN:%.+]] = comb.mux %reset_sink_bits, %true{{.*}}, %en_sink_bits : i1
// CHECK: [[DATA:%.+]] = comb.mux %reset_sink_bits, %c0_i8, %d_sink_bits : i8
// CHECK: seq.firmem.write_port %s[%threadIdx] = [[DATA]], clock %clk_buffer.O enable [[EN]]
hw.module @Hold(in %clk : !seq.clock, in %reset : i1, in %en : i1, in %d : i8, out q : i8, out p : i8) {
  %c0_i8 = hw.constant 0 : i8
  %m = comb.mux %en, %d, %r : i8
  %r = seq.firreg %m clock %clk : i8
  %n = comb.mux %en, %d, %s : i8
  %s = seq.firreg %n clock %clk reset sync %reset, %c0_i8 : i8
  hw.output %r, %s : i8, i8
}

// RUN: foldsim-opt %s --foldsim-fame5="module=Wired threads=4" | FileCheck %s --check-prefix=WIRED

// The hold is below the wiring that narrowed the register to its varying
// bits: the register is still written only when enabled.
// WIRED-LABEL: hw.module @Wired_threaded_Wired(
// WIRED: seq.firmem.write_port %r[%threadIdx] = {{%.+}}, clock %clk_buffer.O enable %en_sink_bits
hw.module @Wired(in %clk : !seq.clock, in %en : i1, in %d : i8, out q : i6) {
  %c0_i2 = hw.constant 0 : i2
  %hi = comb.extract %r from 3 : (i6) -> i3
  %lo = comb.extract %r from 0 : (i6) -> i3
  %wide = comb.concat %hi, %c0_i2, %lo : i3, i2, i3
  %held = hw.bitcast %wide : (i8) -> !hw.array<1xi8>
  %new = hw.array_create %d : i8
  %m = comb.mux %en, %new, %held : !hw.array<1xi8>
  %b = hw.bitcast %m : (!hw.array<1xi8>) -> i8
  %nhi = comb.extract %b from 5 : (i8) -> i3
  %nlo = comb.extract %b from 0 : (i8) -> i3
  %n = comb.concat %nhi, %nlo : i3, i3
  %r = seq.firreg %n clock %clk : i6
  hw.output %r : i6
}

// RUN: foldsim-opt %s --foldsim-fame5="module=Hold threads=4 clock-enable thread-index" | FileCheck %s --check-prefix=CE

// With a clock enable there is no clock gate: the state is written on the host
// clock, enabled when its thread finishes, and the thread is an output.
// CE-LABEL: hw.module @Hold_threaded_Hold(
// CE-SAME:    out threadIdx : i2, out finish : i1)
// CE-NOT:   FireSimFabricClockGate
// CE:       seq.firmem.write_port %r[%threadIdx] = %d_sink_bits, clock %hostClock enable [[EN:%.+]] :
// CE:       [[EN]] = comb.and %en_sink_bits, {{%.+}} : i1
// CE:       hw.output {{.*}}, %threadIdx, {{%.+}} : {{.*}}, i2, i1
