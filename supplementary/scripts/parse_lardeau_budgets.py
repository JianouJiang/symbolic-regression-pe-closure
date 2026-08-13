#!/usr/bin/env python3
"""parse Lardeau curved-backstep LES budget Tecplot BLOCK file.

Goal: produce a per-streamwise-station table of (x/H, beta_clauser, u_tau,
delta99, R^2 of M16 closure vs DNS P/eps), then identify beta_break as the
smallest beta for which R^2 drops below 0.85.

Reads three on-disk files (no downloads):
  - codes/data_processing/_download_cache/2d_budgets/curved_backstep_les/
      curvedbackstep_grid_n.dat          # node grid (I=769, J=161)
      curvedbackstep_budgets_all.dat     # 37 variables on I=768 x J=160 cells
      curvedbackstep_wallquantities.dat  # tau_w, Cp, delta*, theta, delta99
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import os
# Override SCALING_LAW_PROJECT_ROOT env-var if running from a different layout.
SUPP_DIR = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get(
    "SCALING_LAW_PROJECT_ROOT",
    str(SUPP_DIR.parent.parent),
))
DAT_DIR = ROOT / ("codes/data_processing/_download_cache/2d_budgets/"
                  "curved_backstep_les")
RESULTS_DIR = SUPP_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RE_H = 13700.0   # inflow Re based on step height H
NU_OVER_UH = 1.0 / RE_H

# Wall-weighted refit from (../results/refit_wall_weighted.json winner)
ALPHA = (0.11127151106069741,
         0.052205250904985706,
         0.4432637883713678,
         2.887974704849246)


def closure_M16(yp: np.ndarray, a: tuple[float, float, float, float]) -> np.ndarray:
    """tanh^2(a1 y+) / [tanh(a2 y+ - a3) + a4/y+]."""
    a1, a2, a3, a4 = a
    return np.tanh(a1 * yp) ** 2 / (np.tanh(a2 * yp - a3) + a4 / yp)


def parse_tecplot_block(path: Path, n_vars: int, n_cells: int) -> np.ndarray:
    """Read a Tecplot BLOCK file: one variable's I*J values then the next.

    Returns shape (n_vars, n_cells). Skips lines until the first all-numeric
    block of 5 columns starts.
    """
    floats: list[float] = []
    total = n_vars * n_cells
    with path.open() as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith(("TITLE", "ZONE",
                    "variables", "VARIABLES", "STRANDID", "I=", "DATAPACKING",
                    "DT=", "zone")):
                continue
            try:
                parts = [float(x) for x in s.split()]
            except ValueError:
                continue
            floats.extend(parts)
            if len(floats) >= total:
                break
    arr = np.asarray(floats[:total], dtype=np.float64)
    return arr.reshape(n_vars, n_cells)


def main() -> None:
    # 1) Wall quantities
    wq_lines = [l for l in (DAT_DIR / "curvedbackstep_wallquantities.dat").read_text().splitlines()
                if l and not l.startswith("#") and not l.startswith("variables")
                and not l.startswith("TITLE")]
    wq = np.array([[float(t) for t in line.split()] for line in wq_lines])
    # x, y_wall, C_p, tau_w, delta*, theta, delta99
    x_w = wq[:, 0]
    Cp = wq[:, 2]
    tau_w = wq[:, 3]
    dstar = wq[:, 4]
    d99 = wq[:, 6]

    # 2) Budget BLOCK data (37 vars, 768*160 = 122880 cells)
    NI, NJ = 768, 160
    N_VARS = 37
    print(f"Parsing budget BLOCK file ({N_VARS} vars x {NI*NJ} cells)...", flush=True)
    budget = parse_tecplot_block(DAT_DIR / "curvedbackstep_budgets_all.dat",
                                 N_VARS, NI * NJ)
    print(f"  loaded shape={budget.shape}", flush=True)
    # Variables (in order): x/H, y/H, then k(7), uu(7), uv(7), vv(7), ww(7).
    # k block starts at index 2:
    #   prod_k=2, turb_transport_k=3, pres_strain_k=4, pres_diffusion_k=5,
    #   visc_diffusion_k=6, dissipation_k=7, convection_k=8
    # Tecplot BLOCK: i fastest, j slowest → reshape (NJ, NI) then transpose
    # so axis 0 = i (streamwise station), axis 1 = j (wall-normal).
    xH_flat = budget[0].reshape(NJ, NI).T
    yH_flat = budget[1].reshape(NJ, NI).T
    prod_k = budget[2].reshape(NJ, NI).T
    diss_k = budget[7].reshape(NJ, NI).T

    # 3) For each streamwise station i, compute (beta_clauser, R2_M16, u_tau)
    # Wall-quantities file has 768 rows corresponding to the same stations.
    if len(x_w) != NI:
        print(f"WARN: wq rows {len(x_w)} != NI {NI}; will align by nearest x.",
              flush=True)
    # Determine wall-side ordering: y/H should start small (near wall) at j=0.
    # Use station 0 to test:
    if yH_flat[0, 0] > yH_flat[0, -1]:
        # flip so wall is at j=0
        yH_flat = yH_flat[:, ::-1]
        xH_flat = xH_flat[:, ::-1]
        prod_k = prod_k[:, ::-1]
        diss_k = diss_k[:, ::-1]

    # Streamwise pressure gradient from C_p:
    #   p = ½ρU_in² · C_p  →  dp/dx = ½ρU_in² · dCp/dx
    #   β = (δ*/τ_w) · dpe/dx in the same non-dim system
    # tau_w in the file is τ_w/(ρU_in²) (per the Cf = 2*tau_w convention).
    # So β = (δ*/H) · ½ · (dCp/dx) / (tau_w_nondim).
    dCp_dx = np.gradient(Cp, x_w)
    beta_clauser = (dstar / tau_w) * 0.5 * dCp_dx

    # u_tau (per file convention): u_tau/U_in = sqrt(tau_w_nondim)
    u_tau_over_Uin = np.sqrt(np.clip(tau_w, 1e-12, None))

    # Iterate over stations downsampled every 4th (768 → 192 stations)
    STRIDE = 4
    station_ix = list(range(0, NI, STRIDE))

    stations: list[dict] = []
    for ii in station_ix:
        ut = u_tau_over_Uin[ii]
        y_phys = yH_flat[ii] - yH_flat[ii, 0]  # distance from wall
        # y+ = y * u_tau / nu; in H, U_in units: y+ = y * u_tau_over_Uin / (nu/(U_in H))
        yp = y_phys * ut / NU_OVER_UH
        # Skip separated stations (tau_w < 0 → ut imaginary in real flow,
        # but file has tau_w as signed scalar).
        if tau_w[ii] <= 1e-6:
            stations.append({
                "x_over_H": float(x_w[ii]),
                "tau_w": float(tau_w[ii]),
                "u_tau_over_Uin": float(ut),
                "beta_clauser": float(beta_clauser[ii]),
                "delta99": float(d99[ii]),
                "R2_M16": None,
                "P_over_eps_DNS_at_yp10": None,
                "n_pts_used": 0,
                "status": "separated_or_recovering",
            })
            continue
        # Use the inner-layer points: 1 ≤ y+ ≤ min(150, 0.3 d99+)
        d99_p = d99[ii] * ut / NU_OVER_UH
        ymax_p = min(150.0, 0.3 * d99_p)
        mask = (yp >= 1.0) & (yp <= ymax_p) & (diss_k[ii] > 0) & np.isfinite(prod_k[ii])
        if mask.sum() < 5:
            stations.append({
                "x_over_H": float(x_w[ii]),
                "tau_w": float(tau_w[ii]),
                "u_tau_over_Uin": float(ut),
                "beta_clauser": float(beta_clauser[ii]),
                "delta99": float(d99[ii]),
                "R2_M16": None,
                "P_over_eps_DNS_at_yp10": None,
                "n_pts_used": int(mask.sum()),
                "status": "insufficient_points",
            })
            continue
        yp_used = yp[mask]
        P_over_eps_dns = prod_k[ii][mask] / diss_k[ii][mask]
        F_pred = closure_M16(yp_used, ALPHA)
        # R^2 in log-log space (P/eps is positive and spans decades)
        ok = (P_over_eps_dns > 0) & np.isfinite(F_pred) & (F_pred > 0)
        if ok.sum() < 5:
            stations.append({
                "x_over_H": float(x_w[ii]),
                "tau_w": float(tau_w[ii]),
                "u_tau_over_Uin": float(ut),
                "beta_clauser": float(beta_clauser[ii]),
                "delta99": float(d99[ii]),
                "R2_M16": None,
                "P_over_eps_DNS_at_yp10": None,
                "n_pts_used": int(ok.sum()),
                "status": "insufficient_positive_points",
            })
            continue
        y_dns = np.log(P_over_eps_dns[ok])
        y_pre = np.log(F_pred[ok])
        ss_res = float(np.sum((y_dns - y_pre) ** 2))
        ss_tot = float(np.sum((y_dns - y_dns.mean()) ** 2))
        R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        # P/eps at y+ ~ 10 (peak production region) for diagnostic
        idx10 = int(np.argmin(np.abs(yp_used - 10.0)))
        stations.append({
            "x_over_H": float(x_w[ii]),
            "tau_w": float(tau_w[ii]),
            "u_tau_over_Uin": float(ut),
            "beta_clauser": float(beta_clauser[ii]),
            "delta99": float(d99[ii]),
            "R2_M16": float(R2),
            "P_over_eps_DNS_at_yp10": float(P_over_eps_dns[ok][idx10]),
            "n_pts_used": int(ok.sum()),
            "status": "ok",
        })

    # 4) Find beta_break: smallest |beta| > 0 with R^2 < 0.85 in APG regime.
    attached = [s for s in stations if s["status"] == "ok"]
    apg = sorted([s for s in attached if s["beta_clauser"] > 0],
                 key=lambda s: s["beta_clauser"])
    beta_break = None
    for s in apg:
        if s["R2_M16"] is not None and s["R2_M16"] < 0.85:
            beta_break = s["beta_clauser"]
            beta_break_x = s["x_over_H"]
            beta_break_R2 = s["R2_M16"]
            break
    # Also record max-beta successfully validated
    pass_apg = [s for s in apg if s["R2_M16"] is not None and s["R2_M16"] >= 0.85]
    beta_max_pass = max((s["beta_clauser"] for s in pass_apg), default=None)

    # constant-shear check: deviation tracks beta_clauser
    # (the closure assumes ~constant total shear stress). For each station,
    # report tau_w deviation from the local upstream-mean (proxy for total
    # shear stress evolution along x).
    # Simple deviation proxy: |dCp/dx| · δ99 / tau_w (=  ~β scaled by δ99/δ*)

    summary = {
        "dataset": "Lardeau-Leschziner curved backstep LES",
        "reference": ("Lardeau & Leschziner 2011, J. Fluid Mech. 683:172; "
                      "Bentaleb, Lardeau & Leschziner LES of separation from "
                      "rounded step"),
        "Re_H": RE_H,
        "closure": {
            "form": "tanh^2(a1 y+)/[tanh(a2 y+ - a3) + a4/y+]",
            "alpha": list(ALPHA),
            "source": ("../results/refit_wall_weighted.json "
                       "winner (wall_weight=1.0)"),
        },
        "fit_window": "y+ in [1, min(150, 0.3*d99+)]",
        "n_stations_total": len(stations),
        "n_attached": len([s for s in stations if s["status"] == "ok"]),
        "n_apg_attached": len(apg),
        "n_apg_R2_above_085": len(pass_apg),
        "beta_break": beta_break,
        "beta_break_x_over_H": beta_break_x if beta_break is not None else None,
        "beta_break_R2": beta_break_R2 if beta_break is not None else None,
        "beta_max_validated": beta_max_pass,
        "stations": stations,
    }

    out_path = RESULTS_DIR / "track_d_apg_with_budgets.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}", flush=True)

    # 5) Print succinct console summary
    print()
    print("Summary:")
    print(f"  total stations          = {summary['n_stations_total']}")
    print(f"  attached stations       = {summary['n_attached']}")
    print(f"  APG (β > 0) attached    = {summary['n_apg_attached']}")
    print(f"  APG with R² >= 0.85     = {summary['n_apg_R2_above_085']}")
    print(f"  β_max validated         = {summary['beta_max_validated']}")
    print(f"  β_break (first R²<0.85) = {summary['beta_break']}")


if __name__ == "__main__":
    main()
