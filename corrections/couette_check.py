#!/usr/bin/env python3
"""Plane Couette flow: the production-dissipation ratio with the correct budget column.

The article (Sec. IV C, Table V) reported R^2 < 0 for plane Couette flow. Those
values came from reading column 3 of the Lee & Moser (2018) budget files
(turbulent transport) as the dissipation. The dissipation is column 7. This
script downloads the three budget files from the original provider, checks
their SHA-256 digests, and scores both printed formulas with each column on
the article's evaluation mask:

    1 < y+ <= min(150, 0.3 Re_tau),  eps > 1e-6,  P/eps > 0.01

with the nominal Re_tau = 93, 220, 500. Column 3 reproduces the published
values; column 7 gives the corrected ones.

Data: M. Lee and R. D. Moser, "Extreme-scale motions in turbulent plane
Couette flows", J. Fluid Mech. 842, 128-145 (2018). The files are not
redistributed here; they are downloaded to corrections/data/.

Usage (from the repository root):  python3 corrections/couette_check.py
Output: corrections/couette_check.json
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from closure.pe_closure import f_m16, f_pysr  # noqa: E402

BASE_URL = "https://turbulence.oden.utexas.edu/couette2018/data/"
DATA = ROOT / "corrections" / "data"
OUTPUT = ROOT / "corrections" / "couette_check.json"
CASES = {  # nominal Re_tau: (file, SHA-256 of the provider's file)
    93: ("LM_Couette_R0093_020PI_RSTE_k_prof.dat",
         "afb990b13f29294d08d21af87f2d91abe29ca93710d79e21cd407b55af4b07e4"),
    220: ("LM_Couette_R0220_020PI_RSTE_k_prof.dat",
          "d0653dd1557283cc207a9fb66667a4a145285dea8160a1b89a5a9fac8c0ad36c"),
    500: ("LM_Couette_R0500_020PI_RSTE_k_prof.dat",
          "6a5db733e3a56111b6fb4780fd75aabba5f46bac8dcf5771eccd56bd54405ba6"),
}
# Column order of the budget files (from their headers): 0 y/delta, 1 y+, 2 production,
# 3 turbulent transport, 4 viscous transport, 5 pressure strain, 6 pressure transport,
# 7 viscous dissipation, 8 balance.
COLUMNS = {"published (column 3, turbulent transport)": 3, "corrected (column 7, dissipation)": 7}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(name: str, digest: str) -> Path:
    path = DATA / name
    if not path.is_file():
        DATA.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(BASE_URL + name, path)
    if sha256(path) != digest:
        raise RuntimeError(f"{name}: SHA-256 differs from the recorded provider file")
    return path


def r2(target: np.ndarray, model: np.ndarray) -> float:
    return float(1.0 - np.sum((target - model) ** 2) / np.sum((target - target.mean()) ** 2))


def main() -> None:
    record = {"data": "Lee & Moser (2018), plane Couette flow, " + BASE_URL,
              "mask": "1 < y+ <= min(150, 0.3 Re_tau), eps > 1e-6, P/eps > 0.01; nominal Re_tau",
              "cases": {}}
    for re_tau, (name, digest) in CASES.items():
        raw = np.loadtxt(fetch(name, digest), comments="%")
        y_plus, production = raw[:, 1], raw[:, 2]
        case = {"file": name, "sha256": digest}
        for label, column in COLUMNS.items():
            eps = np.abs(raw[:, column])
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = production / eps
            mask = ((y_plus > 1.0) & (y_plus <= min(150.0, 0.3 * re_tau)) & (eps > 1e-6)
                    & (ratio > 0.01) & np.isfinite(ratio))
            y, target = y_plus[mask], ratio[mask]
            case[label] = {"points": int(mask.sum()),
                           "max_P_over_eps": float(target.max()),
                           "y_plus_at_max": float(y[np.argmax(target)]),
                           "R2_recommended": r2(target, f_pysr(y)),
                           "R2_M16": r2(target, f_m16(y))}
        record["cases"][str(re_tau)] = case
    OUTPUT.write_text(json.dumps(record, indent=2) + "\n")
    print(f"{'Re_tau':>6}  {'published R2':>12}  {'corrected R2':>12}  {'M16':>6}  peak P/eps")
    for re_tau, case in record["cases"].items():
        old = case["published (column 3, turbulent transport)"]
        new = case["corrected (column 7, dissipation)"]
        print(f"{re_tau:>6}  {old['R2_recommended']:12.3f}  {new['R2_recommended']:12.3f}  "
              f"{new['R2_M16']:6.3f}  {new['max_P_over_eps']:.2f} at y+ = {new['y_plus_at_max']:.1f}")


if __name__ == "__main__":
    main()
