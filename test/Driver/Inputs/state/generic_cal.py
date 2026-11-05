# generic_cal.py <the foldsim driver, beside the modules> <models.mlir>: a calibration's class factors
# (foldsim_degree.cal_factors): a family that names the class first, else the `generic` model of the class's
# cost features (class_features), else 1. Factors exp(c0 + c . features), clamped to [lo, hi].
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.realpath(sys.argv[1])))
import foldsim_degree as fd
models = sys.argv[2]
f = fd.class_features(models, ['sl', 'root_Bank', 'Top_model'])
for c in sorted(f):
    print(c, ' '.join(f'{k} {f[c][k]:.3f}' for k in fd.GENERIC_FEATURES))
# exp(0.1 + 0.5 * mux_share), at most 1.5; LUTRAM: exp(0) = 1, at least 1.2.
gen = dict(logic=dict(features=['mux_share'], coef=[0.1, 0.5], lo=0.5, hi=1.5),
           lutram=dict(features=['log_lut'], coef=[0.0, 0.0], lo=1.2, hi=3.0))
fd.KERNEL.update(models=models, features=None,
                 calibration=dict(families=[dict(family='slices', match='^sl$', logic=2.0, lutram=0.5)], generic=gen))
for c in ('sl', 'sl_copy1', 'root_Bank', 'nosuch'):
    lg, lr = fd.cal_factors(c)
    print(f'{c}: logic {lg:.3f} lutram {lr:.3f}')
m = f['root_Bank']['mux_share']
print(f"root_Bank expected logic {min(max(math.exp(0.1 + 0.5 * m), 0.5), 1.5):.3f}")
fd.KERNEL.update(calibration=dict(families=[dict(family='slices', match='^sl$', logic=2.0, lutram=0.5)]))
print('no generic:', fd.cal_factors('root_Bank'))
