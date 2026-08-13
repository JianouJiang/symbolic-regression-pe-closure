#!/usr/bin/env python3
"""Generate publication figures for the PoF manuscript with cleaned labels.

Two figures are produced:

1. fig_apg_and_wall_ci.pdf — side-by-side two-panel figure combining
   the broader-APG R^2 vs beta panel (left) with the wall-asymptote
   bootstrap-CI panel (right). Replaces the previous separate
   single-panel figures.

2. fig_rans_deployment.pdf — six-panel 1D Launder-Sharma channel
   deployment figure (U+, k+ at three Re_tau). Replaces the previous
   figure with the same content but rebuilt from the regenerated
   results JSON after the solver blend-width fix.

All labels use publication-clean language (no internal "Track X",
"L_n", "G_n" jargon).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "manuscript" / "figures"
CODE_FIG_DIR = REPO_ROOT / "codes" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CODE_FIG_DIR.mkdir(parents=True, exist_ok=True)

# Data file locations
FORMULA_AUDIT_JSON = (REPO_ROOT / "replay"
                      / "results" / "formula_comparison_exact_replayed.json")
WALL_CI_JSON = REPO_ROOT / "supplementary" / "results" / "wall_slope_ci.json"
RANS_JSON = REPO_ROOT / "codes" / "results" / "track_e_1d_results.json"


def save_both(fig, name):
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png")
    fig.savefig(CODE_FIG_DIR / f"{name}.pdf")
    fig.savefig(CODE_FIG_DIR / f"{name}.png")


def panel_apg(ax):
    """Left panel: descriptive Bobke equilibrium-APG breakdown bracket.

    Recovering/backstep and periodic-hill rows remain supplementary
    out-of-domain stress tests.  In particular, the frozen periodic-hill
    manifest contains no Clauser-beta measurement, so no beta proxy is
    assigned here.
    """
    audit = json.loads(FORMULA_AUDIT_JSON.read_text())
    manifest = json.loads((REPO_ROOT / audit["manifest_path"]).read_text())
    metrics = {row["dataset_id"]: row for row in audit["rows"]}
    selectors = {row["dataset_id"]: row["selector"]
                 for row in manifest["rows"]}

    # Bobke equilibrium APG
    bobke_ids = [key for key in metrics if key.startswith("Bobke_APG_")]
    bobke_pairs = sorted(
        (selectors[key]["beta_range_reported"][1],
         metrics[key]["original_PySR"]["R2"])
        for key in bobke_ids
    )
    bobke_beta = [pair[0] for pair in bobke_pairs]
    bobke_R2 = [pair[1] for pair in bobke_pairs]
    ax.plot(bobke_beta, bobke_R2, "s-", color="C1", lw=1.3, ms=6,
            label="Bobke equilibrium APG TBL")

    ax.axhline(0.85, color="black", ls=":", lw=0.9)
    ax.text(0.98, 0.90, r"$R^2 = 0.85$ threshold", fontsize=7.5,
            color="black", ha="right", transform=ax.transAxes)

    ax.axvspan(2.80, 4.53, color="orange", alpha=0.18,
               label=r"$\beta_{\rm break}$ (equilibrium APG)")
    pred = 1.2
    ax.axvline(pred, color="C2", ls="--", lw=1.0,
               label=rf"constant-stress estimate $\beta\!=\!{pred:.1f}$")

    ax.set_xlim(0.8, 4.8)
    ax.set_ylim(0.70, 0.96)
    ax.set_xlabel(r"Reported $\beta_{\max}$")
    ax.set_ylabel(r"PySR $R^2$ against DNS")
    ax.set_title(r"(a) Equilibrium-APG limit")
    ax.legend(loc="lower left", framealpha=0.95, fontsize=7.0)
    ax.grid(alpha=0.3)


def panel_wall_ci(ax):
    """Right panel: wall-exponent CI vs Re_tau."""
    ci = json.loads(WALL_CI_JSON.read_text())
    Res = [180, 550, 1000, 2000, 5200]
    n_hat = [ci["per_Re_tau"][str(r)]["yp_0.1_to_2.0"]["n_hat"] for r in Res]
    lo = [ci["per_Re_tau"][str(r)]["yp_0.1_to_2.0"]["ci95_lo"] for r in Res]
    hi = [ci["per_Re_tau"][str(r)]["yp_0.1_to_2.0"]["ci95_hi"] for r in Res]
    yerr = np.array([np.array(n_hat) - np.array(lo),
                     np.array(hi) - np.array(n_hat)])
    ax.errorbar(Res, n_hat, yerr=yerr, fmt="o-", color="C0", capsize=3,
                lw=1.2, ms=5, label="DNS bootstrap 95% CI")
    ax.axhline(3.0, color="black", ls="--", lw=1.0,
               label=r"M16 benchmark $n=3$")
    ax.fill_between([100, 7000], 2.9, 3.1, color="grey", alpha=0.18,
                    label=r"Nominal tolerance $[2.9, 3.1]$")
    ax.fill_between([100, 7000], 3.1, 3.15, color="C2", alpha=0.18,
                    label=r"Extended tolerance $[2.9, 3.15]$")
    ax.set_xscale("log")
    ax.set_xlim(120, 7000)
    ax.set_ylim(2.85, 3.20)
    ax.set_xlabel(r"$\mathrm{Re}_\tau$")
    ax.set_ylabel(r"Wall exponent $n$")
    ax.set_title(r"(b) Wall-exponent audit")
    ax.legend(loc="lower right", framealpha=0.95, fontsize=7.5)
    ax.grid(alpha=0.3)


def figure_apg_and_wall_ci():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    panel_apg(axes[0])
    panel_wall_ci(axes[1])
    fig.tight_layout()
    save_both(fig, "fig_apg_and_wall_ci")
    plt.close(fig)
    return "fig_apg_and_wall_ci"


def figure_rans_deployment():
    """Six-panel U+ / k+ figure at Re_tau in {180, 1000, 5200}."""
    d = json.loads(RANS_JSON.read_text())

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.2), sharex=True)
    titles = [180, 1000, 5200]

    for col, case in enumerate(d["cases"]):
        re_tau = case["re_tau"]
        prof = case["profiles"]
        yp_dns = np.asarray(prof["yp_dns"])
        Up_dns = np.asarray(prof["Up_dns"])
        kp_dns = np.asarray(prof["kp_dns"])
        yp_sim = np.asarray(prof["yp_sim"])
        Up_base = np.asarray(prof["Up_base"])
        Up_m16 = np.asarray(prof["Up_m16"])
        kp_base = np.asarray(prof["kp_base"])
        kp_m16 = np.asarray(prof["kp_m16"])

        # U+ panel
        ax = axes[0, col]
        ax.semilogx(yp_dns, Up_dns, "k-", lw=1.2, label="DNS (Lee--Moser)")
        ax.semilogx(yp_sim, Up_base, "b--", lw=1.2,
                    label="Baseline $k$--$\\varepsilon$")
        ax.semilogx(yp_sim, Up_m16, "r-", lw=1.4, label="M16-corrected")
        ax.set_title(rf"Channel $\mathrm{{Re}}_\tau = {re_tau}$")
        ax.set_ylabel(r"$U^+$") if col == 0 else None
        ax.set_xlim(0.5, max(yp_dns.max(), yp_sim.max()) * 1.1)
        ax.set_ylim(0, max(Up_dns.max(), Up_base.max(), Up_m16.max()) * 1.08)
        ax.grid(alpha=0.3)
        # Annotate metrics inside panel
        cf_b = case["Cf_rel_err_base"] * 100
        cf_m = case["Cf_rel_err_m16"] * 100
        umr = case["U_plus_rmse_reduction_pct"]
        ax.text(0.97, 0.06,
                f"$C_f$ rel-err base/M16: {cf_b:.0f}%/{cf_m:.0f}%\n"
                f"$U^+$ RMSE red.: {umr:.1f}%",
                transform=ax.transAxes, fontsize=7,
                ha="right", va="bottom",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
        if col == 0:
            ax.legend(loc="upper left", fontsize=7, framealpha=0.95)

        # k+ panel
        ax = axes[1, col]
        ax.semilogx(yp_dns, kp_dns, "k-", lw=1.2)
        ax.semilogx(yp_sim, kp_base, "b--", lw=1.2)
        ax.semilogx(yp_sim, kp_m16, "r-", lw=1.4)
        ax.set_xlabel(r"$y^+$")
        ax.set_ylabel(r"$k^+$") if col == 0 else None
        ax.set_xlim(0.5, max(yp_dns.max(), yp_sim.max()) * 1.1)
        kp_max = max(kp_dns.max(), kp_base.max(), kp_m16.max()) * 1.08
        ax.set_ylim(0, kp_max)
        ax.grid(alpha=0.3)
        kmr = case["k_plus_rmse_reduction_pct"]
        ax.text(0.97, 0.94,
                f"$k^+$ RMSE red.: {kmr:.1f}%",
                transform=ax.transAxes, fontsize=7,
                ha="right", va="top",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))

    fig.tight_layout()
    save_both(fig, "fig_rans_deployment")
    plt.close(fig)
    return "fig_rans_deployment"


if __name__ == "__main__":
    n1 = figure_apg_and_wall_ci()
    print(f"Saved: {n1}.{{pdf,png}}")
    if RANS_JSON.exists():
        n2 = figure_rans_deployment()
        print(f"Saved: {n2}.{{pdf,png}}")
    else:
        print(f"Skipped RANS figure: {RANS_JSON} not present yet")
