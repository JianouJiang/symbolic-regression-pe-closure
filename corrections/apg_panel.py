#!/usr/bin/env python3
"""Corrected Fig. 3(d): adverse-pressure-gradient boundary layers.

Panel (d) of the published Fig. 3 shows five zero-pressure-gradient boundary-layer
profiles (a subset of panel b): the plotting script fell back to those profiles when
the adverse-pressure-gradient files were missing. This script draws the intended
panel from the adverse-pressure-gradient data of Bobke et al. (2017): for each of the
five series (b1n, m13n, b2n, m16n, m18n), the station with the largest Clauser
parameter beta, with y+ = y u_tau / nu and P/eps = Pk / |Dk| as in
records/manifest/full_suite_evaluation.py, over 0.5 < y+ <= 80.

The numbers quoted in the article for these series (Sec. IV C, Table V, Figs. 4c and
5a) were computed from the same data and are unchanged.

Data: P. Bobke, R. Vinuesa, R. Orlu and P. Schlatter, J. Fluid Mech. 820, 667-692
(2017); APG.mat (SHA-256 bf87b389...), staged at records/manifest/source_cache/ as
described in the README.

Usage (from the repository root):  python3 corrections/apg_panel.py [--apg PATH]
Outputs: corrections/apg_panel.json, corrections/fig3d_corrected.png
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from closure.pe_closure import f_pysr  # noqa: E402

APG_SHA256 = "bf87b389ca359747ab400317c774cbece3ab918b2224f8bb1bd7ca24f947767a"
SERIES = ["b1n", "m13n", "b2n", "m16n", "m18n"]
OUT = ROOT / "corrections"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(value) -> float:
    return float(np.asarray(value).ravel()[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apg", type=Path, default=ROOT / "records" / "manifest" / "source_cache" / "APG.mat")
    args = parser.parse_args()
    if sha256(args.apg) != APG_SHA256:
        raise RuntimeError("APG.mat differs from the file used in the article")
    curves = {}
    for case in SERIES:
        stations = loadmat(args.apg, variable_names=[case])[case]
        beta = np.array([scalar(stations[0, i]["beta"]) for i in range(stations.shape[1])])
        index = int(np.argmax(beta))
        station = stations[0, index]
        u_tau, nu = scalar(station["ut"]), scalar(station["nu"])
        y_plus = np.asarray(station["y"], dtype=float).ravel() * u_tau / nu
        production = np.asarray(station["Pk"], dtype=float).ravel()
        eps = np.abs(np.asarray(station["Dk"], dtype=float).ravel())
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = production / eps
        mask = (y_plus > 0.5) & (y_plus <= 80.0) & (eps > 1e-6) & (ratio > 0.01) & np.isfinite(ratio)
        curves[case] = {"station_index": index, "x": scalar(station["x"]), "beta": float(beta[index]),
                        "Re_tau": scalar(station["Ret"]), "y_plus": y_plus[mask].tolist(),
                        "P_over_eps": ratio[mask].tolist()}
    (OUT / "apg_panel.json").write_text(json.dumps(
        {"data": "Bobke et al. (2017), APG.mat, SHA-256 " + APG_SHA256,
         "station": "largest beta in each series", "series": curves}, indent=1) + "\n")

    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    colours = plt.get_cmap("Purples")(np.linspace(0.4, 0.95, len(SERIES)))
    for colour, case in zip(colours, sorted(SERIES, key=lambda c: curves[c]["beta"])):
        c = curves[case]
        ax.plot(c["y_plus"], c["P_over_eps"], color=colour, lw=1.4,
                label=rf"{case}, $\beta={c['beta']:.2f}$")
    y_fit = np.linspace(1.0, 90.0, 400)
    ax.plot(y_fit, f_pysr(y_fit), color="k", lw=1.6, label="Recommended expression")
    ax.axhline(1.0, color="grey", ls="--", lw=0.5)
    ax.set(xscale="log", xlim=(1, 150), ylim=(0, 2.5), xlabel=r"$y^+$", ylabel=r"$P/\varepsilon$",
           title="(d) APG boundary layers (corrected)")
    ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(OUT / "fig3d_corrected.png", dpi=200)
    for case in SERIES:
        c = curves[case]
        print(f"{case}: station {c['station_index']} at x = {c['x']:.1f}, beta = {c['beta']:.3f}, "
              f"Re_tau = {c['Re_tau']:.0f}, peak P/eps = {max(c['P_over_eps']):.2f}")


if __name__ == "__main__":
    main()
