# TinyTOML: the TOML reader of Vortex's ci/gen_config.py (foldsim/one-master 6e5db10cd), verbatim, for
# gen_platform.py's card descriptors: these builds run under whatever python3 the tool environment provides
# (FireSim's conda env is 3.10, before tomllib), and descriptors stay within the subset it reads.
from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional, Tuple


class TinyTOML:
  def parse(self, content: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    cursor: Dict[str, Any] = root

    for raw in content.splitlines():
      # Strip comments (works for your configs; none have '#' inside strings)
      line = raw.split("#", 1)[0].strip()
      if not line:
        continue

      if line.startswith("[[") and line.endswith("]]"):
        key = line[2:-2].strip()
        arr = root.get(key)
        if arr is None:
          arr = []
          root[key] = arr
        if not isinstance(arr, list):
          raise ValueError(f"TOML key {key!r} used as both table and array-of-tables")
        tbl: Dict[str, Any] = {}
        arr.append(tbl)
        cursor = tbl
        continue

      if line.startswith("[") and line.endswith("]"):
        key = line[1:-1].strip()
        tbl = root.get(key)
        if tbl is None:
          tbl = {}
          root[key] = tbl
        if not isinstance(tbl, dict):
          raise ValueError(f"TOML key {key!r} used as both scalar and table")
        cursor = tbl
        continue

      if "=" in line:
        k, v = line.split("=", 1)
        key = k.strip()
        val = v.strip()
        cursor[key] = self._parse_value(val)
        continue

      raise ValueError(f"Unrecognized TOML line: {raw!r}")
    return root

  def _parse_value(self, val: str) -> Any:
    if val == "true": return True
    if val == "false": return False
    try:
      return ast.literal_eval(val)
    except (ValueError, SyntaxError):
      return val
