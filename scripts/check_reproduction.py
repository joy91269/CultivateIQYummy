"""Check that a rerun reproduces the committed results.

    bash run_all.sh --analysis
    python scripts/check_reproduction.py

Compares every table in outputs/tables/ with the version committed in git (HEAD). Numbers may differ in the last
digits across platforms, so numeric columns are compared with a small tolerance; text columns must match exactly.
Figures are not compared (fonts and image encoders differ between machines).
"""
from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs/tables"
RTOL, ATOL = 1e-4, 1e-6


def committed(path: Path) -> pd.DataFrame | None:
    rel = path.relative_to(ROOT).as_posix()
    out = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True)
    if out.returncode != 0:
        return None
    return pd.read_csv(io.BytesIO(out.stdout))


def compare(new: pd.DataFrame, old: pd.DataFrame) -> str | None:
    if list(new.columns) != list(old.columns):
        return "columns differ"
    if len(new) != len(old):
        return f"{len(new)} rows instead of {len(old)}"
    for col in new.columns:
        a, b = new[col], old[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            if not np.allclose(a.to_numpy(float), b.to_numpy(float), rtol=RTOL, atol=ATOL, equal_nan=True):
                return f"values differ in column '{col}'"
        elif not (a.fillna("").astype(str) == b.fillna("").astype(str)).all():
            return f"text differs in column '{col}'"
    return None


def main() -> int:
    failed = 0
    files = sorted(TABLES.glob("*.csv"))
    for path in files:
        old = committed(path)
        if old is None:
            print(f"  new     {path.name} (not in git)")
            continue
        problem = compare(pd.read_csv(path), old)
        if problem:
            failed += 1
            print(f"  DIFFERS {path.name}: {problem}")
        else:
            print(f"  same    {path.name}")
    print(f"\n{len(files) - failed} of {len(files)} tables reproduce the committed results.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
