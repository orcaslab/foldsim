# Timing screen scripts

Run `foldsim-estimate-timing` on an elaborated design and compare it with
Vivado. `foldsim-opt` comes from the `PATH`, or from `FOLDSIM_OPT`.

| Script | What |
|---|---|
| `screen.sh <design.mlir\|mapped.mlirbc> <outdir> [options]` | Map to LUT6 with `foldsim-timing-map`, then screen. Writes `endpoints.csv`, `instances.csv` and `report.txt`; a `.mlirbc` input is taken as already mapped |
| `summarize.py <outdir>` | End points by RTL unit, optionally beside Vivado's (`--ep`) |
| `validate.py <outdir> <ep.csv>` | Agreement per register with a Vivado end point table (`slack,levels,logic,net,start,end`) |
| `retime_split.py <outdir> <ep.csv> MODULE=PREFIX` | The same agreement, split by whether Vivado retimed the register's paths |

```bash
utils/timing/screen.sh gen/design.mlir runs/full
utils/timing/screen.sh runs/full/mapped.mlirbc runs/socket \
    top=VX_socket__lift black-boxes=VX_core__lift
python3 utils/timing/summarize.py runs/full --min-levels 12
```

Every `foldsim-opt` run is capped at 32 GB in a `systemd-run` scope.
