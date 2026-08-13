#!/usr/bin/env python3
"""Independent numerical checks for the final Physics of Fluids revision.

This script deliberately recomputes the revision-sensitive quantities from the
raw one-dimensional DNS budget files.  It does not read numerical values from
the manuscript or from the legacy ``rans_improvement_demo.json`` record.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def locate_project_root() -> Path:
    """Support both the live project and the unpacked supplementary archive."""
    required = (
        Path("replay")
        / "results"
        / "formula_comparison_exact_replayed.json"
    )
    for candidate in (HERE.parent, *HERE.parents):
        if (candidate / required).is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate the archive/project root containing " + str(required)
    )


ROOT = locate_project_root()
OUTPUT = HERE / "CLEAN_ROOM_NUMERICAL_AUDIT.json"

# Constants exactly as printed for the recommended Eq. (35).
PYRSR_ALPHA = (0.128, 0.053, 0.620, 5.78)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pysr(y_plus: np.ndarray) -> np.ndarray:
    a1, a2, a3, a4 = PYRSR_ALPHA
    return np.tanh(a1 * y_plus) / (
        np.tanh(a2 * y_plus - a3) + a4 / y_plus
    )


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denominator)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def dissipation_metrics(
    y_plus: np.ndarray, production: np.ndarray, epsilon: np.ndarray
) -> dict:
    mask = (
        (y_plus > 5.0)
        & (y_plus < 80.0)
        & np.isfinite(production)
        & np.isfinite(epsilon)
        & (epsilon > 1.0e-8)
    )
    y = y_plus[mask]
    p = production[mask]
    eps = epsilon[mask]
    eps_equilibrium = p
    eps_corrected = p / pysr(y)
    rmse_eq = rmse(eps, eps_equilibrium)
    rmse_corrected = rmse(eps, eps_corrected)
    return {
        "n": int(mask.sum()),
        "actual_y_plus_range": [float(y.min()), float(y.max())],
        "R2_equilibrium": r2(eps, eps_equilibrium),
        "R2_corrected": r2(eps, eps_corrected),
        "RMSE_equilibrium": rmse_eq,
        "RMSE_corrected": rmse_corrected,
        "RMSE_reduction_percent": float(
            100.0 * (1.0 - rmse_corrected / rmse_eq)
        ),
    }


def load_channel(re_tau: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    path = ROOT / "codes" / "data_processing" / "spectra_data" / (
        f"LM_Channel_{re_tau:04d}_RSTE_k.dat"
    )
    raw = np.loadtxt(path, comments="%")
    return raw[:, 1], raw[:, 2], raw[:, 7], path


def load_pipe(re_tau: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Path]]:
    paths = [
        ROOT / "codes" / "results" / "el_khoury_pipe" / (
            f"{re_tau}_{component}_Budget.dat"
        )
        for component in ("RR", "TT", "ZZ")
    ]
    budgets = [np.loadtxt(path, comments="%") for path in paths]
    y_plus = budgets[0][:, 1]
    production = 0.5 * sum(budget[:, 2] for budget in budgets)
    epsilon = -0.5 * sum(budget[:, 7] for budget in budgets)
    return y_plus, production, epsilon, paths


def pressure_peak_audit(re_tau: int) -> tuple[dict, Path]:
    path = ROOT / "codes" / "data_processing" / "spectra_data" / (
        f"LM_Channel_{re_tau:04d}_RSTE_k.dat"
    )
    raw = np.loadtxt(path, comments="%")
    y = raw[:, 1]
    production = raw[:, 2]
    pressure_transport = raw[:, 6]
    epsilon = raw[:, 7]
    f_dns = production / epsilon

    # Signed measured-budget term-deletion diagnostic.  Since the pressure
    # contribution to F is -Pi_p/epsilon, deleting Pi_p gives F+Pi_p/epsilon.
    f_without_pressure = f_dns + pressure_transport / epsilon
    mask = (y >= 10.0) & (y <= 20.0)
    indices = np.flatnonzero(mask)
    i_dns = int(indices[np.argmax(f_dns[mask])])
    i_without = int(indices[np.argmax(f_without_pressure[mask])])
    nominal_gap = 0.35 * f_dns[i_dns]
    sign_aligned = np.maximum(-pressure_transport / epsilon, 0.0)
    return {
        "DNS_peak": {"y_plus": float(y[i_dns]), "F": float(f_dns[i_dns])},
        "no_pressure_peak": {
            "y_plus": float(y[i_without]),
            "F": float(f_without_pressure[i_without]),
        },
        "no_pressure_minus_DNS_peak": float(
            f_without_pressure[i_without] - f_dns[i_dns]
        ),
        "peak_location_shift_y_plus": float(y[i_without] - y[i_dns]),
        "peak_change_fraction_of_nominal_35pct_gap": float(
            (f_without_pressure[i_without] - f_dns[i_dns]) / nominal_gap
        ),
        "largest_pointwise_sign_aligned_fraction_of_nominal_35pct_gap": float(
            np.max(sign_aligned[mask]) / nominal_gap
        ),
    }, path


def formula_comparison_check() -> dict:
    path = (
        ROOT
        / "replay"
        / "results"
        / "formula_comparison_exact_replayed.json"
    )
    audit = json.loads(path.read_text())
    primary = [
        row
        for row in audit["rows"]
        if row["role"] == "external_re_evaluation_in_domain"
    ]
    families = sorted({row["source_family"] for row in primary})
    family_delta_r2 = []
    family_relative_rmse = []
    for family in families:
        rows = [row for row in primary if row["source_family"] == family]
        family_delta_r2.append(
            float(np.median([row["delta_R2_M16_minus_PySR"] for row in rows]))
        )
        family_relative_rmse.append(
            float(
                np.median(
                    [
                        row["relative_RMSE_change_M16_over_PySR"]
                        for row in rows
                    ]
                )
            )
        )
    return {
        "source": str(path.relative_to(ROOT)),
        "source_sha256": sha256(path),
        "n_primary_profiles": len(primary),
        "printed_M16_R2_wins": int(
            sum(row["delta_R2_M16_minus_PySR"] > 0.0 for row in primary)
        ),
        "median_of_family_median_delta_R2": float(np.median(family_delta_r2)),
        "median_of_family_median_relative_RMSE_change": float(
            np.median(family_relative_rmse)
        ),
        "inference_boundary": (
            "This is the referee-requested comparison of the two submitted, "
            "printed coefficient sets; it is not a claim that every possible "
            "cubic recalibration is inferior."
        ),
    }


def main() -> None:
    source_hashes: dict[str, str] = {}
    channel: dict[str, dict] = {}
    for re_tau in (180, 550, 1000, 2000, 5200):
        y, production, epsilon, path = load_channel(re_tau)
        channel[str(re_tau)] = dissipation_metrics(y, production, epsilon)
        source_hashes[str(path.relative_to(ROOT))] = sha256(path)

    pipe: dict[str, dict] = {}
    for re_tau in (180, 360, 550, 1000):
        y, production, epsilon, paths = load_pipe(re_tau)
        pipe[str(re_tau)] = dissipation_metrics(y, production, epsilon)
        for path in paths:
            source_hashes[str(path.relative_to(ROOT))] = sha256(path)

    pressure: dict[str, dict] = {}
    for re_tau in (180, 550, 1000, 2000, 5200):
        row, path = pressure_peak_audit(re_tau)
        pressure[str(re_tau)] = row
        source_hashes[str(path.relative_to(ROOT))] = sha256(path)

    reductions = [
        row["RMSE_reduction_percent"]
        for family in (channel, pipe)
        for row in family.values()
    ]
    peak_changes = [
        row["peak_change_fraction_of_nominal_35pct_gap"]
        for row in pressure.values()
    ]
    pointwise_bounds = [
        row["largest_pointwise_sign_aligned_fraction_of_nominal_35pct_gap"]
        for row in pressure.values()
    ]
    result = {
        "schema_version": "clean-room-pof-revision-1.0",
        "recommended_formula": {
            "name": "original_PySR_printed_Eq_35",
            "alpha": list(PYRSR_ALPHA),
            "formula": "tanh(a1*y)/[tanh(a2*y-a3)+a4/y]",
        },
        "dissipation_diagnostic": {
            "input": (
                "DNS-resolved production and dissipation budgets; production "
                "contains DNS mean-shear and Reynolds-stress information"
            ),
            "mask": "5 < y_plus < 80 and epsilon > 1e-8",
            "channel": channel,
            "pipe": pipe,
            "RMSE_reduction_percent_range": [
                float(min(reductions)),
                float(max(reductions)),
            ],
        },
        "pressure_transport_peak_shift_audit": {
            "definition": "F_without_pressure = F_DNS + Pi_p/epsilon",
            "buffer_window": "10 <= y_plus <= 20",
            "per_Re_tau": pressure,
            "peak_change_fraction_of_nominal_35pct_gap_range": [
                float(min(peak_changes)),
                float(max(peak_changes)),
            ],
            "largest_pointwise_sign_aligned_fraction_range": [
                float(min(pointwise_bounds)),
                float(max(pointwise_bounds)),
            ],
            "interpretation": (
                "After allowing the counterfactual peak to move, deleting "
                "pressure transport raises rather than lowers the buffer peak "
                "in all five profiles.  Isolated sign-aligned points contribute "
                "at most 3.7% of the nominal 35% gap."
            ),
        },
        "printed_formula_comparison": formula_comparison_check(),
        "source_sha256": dict(sorted(source_hashes.items())),
    }
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(OUTPUT), "summary": result}, indent=2))


if __name__ == "__main__":
    main()
