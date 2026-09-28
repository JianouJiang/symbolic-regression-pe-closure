#!/usr/bin/env python3
"""A-priori dissipation estimate eps = P/F(y+) by wall-distance region.

The article's 88-95 % reduction of the dissipation-prediction RMSE (Sec. V B,
Table VI) is measured over 5 < y+ < 80; the abstract and introduction quote it
without that range. This script repeats the diagnostic of
audit/recompute_clean_room_metrics.py (same loaders, same channel and pipe
budgets, DNS production) with the article's evaluation mask (eps > 1e-6,
P/eps > 0.01) over the published window and over the whole inner-layer domain
1 < y+ <= min(150, 0.3 Re_tau), split at y+ = 5, for the recommended expression
and for M16.

Usage (from the repository root, with the budgets staged as described in the
README):  python3 corrections/dissipation_by_region.py
Output: corrections/dissipation_by_region.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "audit"))
from closure.pe_closure import f_m16, f_pysr  # noqa: E402
from recompute_clean_room_metrics import load_channel, load_pipe  # noqa: E402

OUTPUT = ROOT / "corrections" / "dissipation_by_region.json"
REGIONS = {
    "published window 5 < y+ < 80": lambda y, re: (y > 5.0) & (y < 80.0),
    "whole domain 1 < y+ <= min(150, 0.3 Re_tau)": lambda y, re: (y > 1.0) & (y <= min(150.0, 0.3 * re)),
    "viscous sublayer 1 < y+ <= 5": lambda y, re: (y > 1.0) & (y <= 5.0),
    "domain above y+ = 5": lambda y, re: (y > 5.0) & (y <= min(150.0, 0.3 * re)),
}


def reduction(y: np.ndarray, production: np.ndarray, eps: np.ndarray, closure) -> float:
    rmse = lambda model: np.sqrt(np.mean((eps - model) ** 2))
    return float(100.0 * (1.0 - rmse(production / closure(y)) / rmse(production)))


def main() -> None:
    profiles = [("channel", re, load_channel(re)) for re in (180, 550, 1000, 2000, 5200)]
    profiles += [("pipe", re, load_pipe(re)) for re in (180, 360, 550, 1000)]
    record = {"diagnostic": "RMSE reduction (%) of eps = P/F(y+) against eps = P, DNS production and dissipation",
              "profiles": {}, "ranges": {}}
    for flow, re_tau, (y_plus, production, eps, _) in profiles:
        row = {}
        for region, select in REGIONS.items():
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = production / eps
            # the article's evaluation mask for P/eps: eps > 1e-6 and P/eps > 0.01
            mask = select(y_plus, re_tau) & np.isfinite(ratio) & (eps > 1e-6) & (ratio > 0.01)
            y, p, e = y_plus[mask], production[mask], eps[mask]
            row[region] = {"points": int(mask.sum()),
                           "recommended": reduction(y, p, e, f_pysr),
                           "M16": reduction(y, p, e, f_m16)}
        record["profiles"][f"{flow} {re_tau}"] = row
    for region in REGIONS:
        for closure in ("recommended", "M16"):
            values = [row[region][closure] for row in record["profiles"].values()]
            record["ranges"][f"{region}, {closure}"] = [min(values), max(values)]
    OUTPUT.write_text(json.dumps(record, indent=2) + "\n")
    for key, (low, high) in record["ranges"].items():
        print(f"{key:62s} {low:5.1f} to {high:5.1f} %")


if __name__ == "__main__":
    main()
