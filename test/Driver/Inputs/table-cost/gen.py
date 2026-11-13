# gen.py: a simulator level with two constant tables in the folded simulator's print form (an i1 array's elements
# bare: `[false, true, ...]`): a 1,024 x i1 slot table with one run of ones (entries 100..139) and an all-zero
# 8,192 x i1 one, each read at a non-constant index.
import sys
t1 = ', '.join('true' if 100 <= k < 140 else 'false' for k in range(1024))
t2 = ', '.join('false' for _ in range(8192))
print('module {')
print('  hw.module @foldsim_sim(in %p : i10, in %q : i13, out o0 : i1, out o1 : i1) {')
print(f'    %t1 = hw.aggregate_constant [{t1}] : !hw.array<1024xi1>')
print('    %r1 = hw.array_get %t1[%p] : !hw.array<1024xi1>, i10')
print(f'    %t2 = hw.aggregate_constant [{t2}] : !hw.array<8192xi1>')
print('    %r2 = hw.array_get %t2[%q] : !hw.array<8192xi1>, i13')
print('    hw.output %r1, %r2 : i1, i1')
print('  }')
print('}')
