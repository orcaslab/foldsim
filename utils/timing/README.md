# Timing screen scripts

Run `foldsim-estimate-timing` on an elaborated design and compare it with
Vivado. `foldsim-opt` comes from the `PATH`, or from `FOLDSIM_OPT`.

| Script | What |
|---|---|
| `screen.sh <design.mlir\|mapped.mlirbc> <outdir> [options]` | Map to LUT6 with `foldsim-timing-map`, then screen. Writes `endpoints.csv`, `instances.csv` and `report.txt`; a `.mlirbc` input is taken as already mapped |
| `summarize.py <outdir>` | End points by RTL unit, optionally beside Vivado's (`--ep`) |
| `validate.py <outdir> <ep.csv>` | Agreement per register with a Vivado end point table (`slack,levels,logic,net,start,end`) |
| `retime_split.py <outdir> <ep.csv> MODULE=PREFIX` | The same agreement, split by whether Vivado retimed the register's paths |

`summarize.py`, `validate.py` and `retime_split.py` compare the `levels`
column by default, or `retimed` (`--metric retimed`, or a last argument for
`retime_split.py`) for a screen run with `retime-levels=N`. With
`retime-levels=10`, end points at `retimed` ≥ 12 are a short list of fix
candidates that Vivado's retiming is unlikely to absorb; plain `levels` stays
the ranking.

The screen counts levels for the V80 (`family=versal`), or for the U55C
with `family=virtexuplusHBM`. The two differ only in their DSPs.

```bash
utils/timing/screen.sh gen/design.mlir runs/full
utils/timing/screen.sh runs/full/mapped.mlirbc runs/socket \
    top=VX_socket__lift black-boxes=VX_core__lift
python3 utils/timing/summarize.py runs/full --min-levels 12
```

Every `foldsim-opt` run is capped at 32 GB in a `systemd-run` scope.
