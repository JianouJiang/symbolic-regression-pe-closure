#!/usr/bin/env python3
"""Reynolds-resolved pressure-transport audit for Lee--Moser training DNS."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
NODE = HERE
ROOT = HERE.parents[1]
DEFAULT_CONFIG = HERE / "audit_config.json"
DEFAULT_OUTPUT = NODE / "results" / "pressure_transport_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_budget(re_tau: int) -> tuple[dict[str, np.ndarray], Path]:
    path = (
        ROOT
        / "codes"
        / "data_processing"
        / "spectra_data"
        / f"LM_Channel_{re_tau:04d}_RSTE_k.dat"
    )
    raw = np.loadtxt(path, comments="%")
    return {
        "y_delta": raw[:, 0],
        "y_plus": raw[:, 1],
        "production": raw[:, 2],
        "turb_transport": raw[:, 3],
        "visc_transport": raw[:, 4],
        "pressure_strain": raw[:, 5],
        "pressure_transport": raw[:, 6],
        "epsilon": raw[:, 7],
        "reported_balance": raw[:, 8],
    }, path


def fraction(condition: np.ndarray, eligible: np.ndarray | None = None) -> float:
    if eligible is None:
        eligible = np.ones(condition.shape, dtype=bool)
    if not np.any(eligible):
        return float("nan")
    return float(np.mean(condition[eligible]))


def layer_metrics(data: dict[str, np.ndarray], bounds: list[float]) -> dict:
    low, high = (float(bounds[0]), float(bounds[1]))
    y = data["y_plus"]
    mask = (y >= low) & (y <= high)
    if mask.sum() < 2:
        return {
            "requested_y_plus_range": [low, high],
            "status": "insufficient points",
            "n": int(mask.sum()),
        }

    y = y[mask]
    turb = data["turb_transport"][mask]
    visc = data["visc_transport"][mask]
    pressure = data["pressure_transport"][mask]
    denom = np.abs(turb) + np.abs(visc)
    valid = denom > np.finfo(float).tiny
    q = np.abs(pressure[valid]) / denom[valid]
    q_signed = pressure[valid] / denom[valid]
    net_diff = turb + visc
    nonzero_pressure = pressure != 0
    nonzero_turb = (pressure != 0) & (turb != 0)
    nonzero_net = (pressure != 0) & (net_diff != 0)

    order = np.argsort(y)
    ys = y[order]
    pressure_s = pressure[order]
    turb_s = turb[order]
    visc_s = visc[order]
    l1_denominator = np.trapezoid(np.abs(turb_s) + np.abs(visc_s), ys)
    l1_ratio = (
        np.trapezoid(np.abs(pressure_s), ys) / l1_denominator
        if l1_denominator > 0
        else float("nan")
    )

    return {
        "requested_y_plus_range": [low, high],
        "actual_y_plus_range": [float(y.min()), float(y.max())],
        "status": "ok",
        "n": int(mask.sum()),
        "pointwise_abs_ratio_median": float(np.median(q)),
        "pointwise_abs_ratio_IQR": np.percentile(q, [25, 75]).tolist(),
        "pointwise_signed_ratio_median": float(np.median(q_signed)),
        "integrated_L1_ratio": float(l1_ratio),
        "pressure_positive_fraction": fraction(pressure > 0, nonzero_pressure),
        "pressure_negative_fraction": fraction(pressure < 0, nonzero_pressure),
        "pressure_opposes_turbulent_transport_fraction": fraction(
            pressure * turb < 0, nonzero_turb
        ),
        "pressure_opposes_turbulent_plus_viscous_fraction": fraction(
            pressure * net_diff < 0, nonzero_net
        ),
    }


def point_diagnostic(data: dict[str, np.ndarray], index: int, deficit_fraction: float) -> dict:
    y = data["y_plus"]
    eps = data["epsilon"]
    production = data["production"]
    pressure = data["pressure_transport"]
    f_dns = production / eps
    delta_pressure = -pressure / eps
    reported_deficit = deficit_fraction * f_dns[index]
    signed_fraction = (
        delta_pressure[index] / reported_deficit
        if abs(reported_deficit) > np.finfo(float).tiny
        else float("nan")
    )
    return {
        "y_plus": float(y[index]),
        "F_DNS": float(f_dns[index]),
        "Pi_p": float(pressure[index]),
        "epsilon": float(eps[index]),
        "delta_F_from_pressure_minus_Pi_over_epsilon": float(delta_pressure[index]),
        "reported_35pct_peak_deficit_normalizer": float(reported_deficit),
        "signed_fraction_of_reported_deficit": float(signed_fraction),
        "sign_aligned_fraction_clipped_0_1": float(np.clip(signed_fraction, 0.0, 1.0)),
    }


def aggregate_equal_profile(per_re: dict[str, dict], layer_names: list[str]) -> dict:
    output: dict[str, dict] = {}
    for layer in layer_names:
        rows = [per_re[key]["layers"][layer] for key in per_re]
        rows = [row for row in rows if row.get("status") == "ok"]
        medians = np.array([row["pointwise_abs_ratio_median"] for row in rows])
        integrals = np.array([row["integrated_L1_ratio"] for row in rows])
        opposition = np.array(
            [row["pressure_opposes_turbulent_transport_fraction"] for row in rows]
        )
        output[layer] = {
            "aggregation_unit": "Re_tau profile (equal weight; no grid pooling)",
            "n_profiles": len(rows),
            "mean_of_profile_medians": float(np.mean(medians)),
            "median_of_profile_medians": float(np.median(medians)),
            "range_of_profile_medians": [float(medians.min()), float(medians.max())],
            "mean_integrated_L1_ratio": float(np.mean(integrals)),
            "median_integrated_L1_ratio": float(np.median(integrals)),
            "mean_opposition_fraction_to_turbulent_transport": float(
                np.mean(opposition)
            ),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    layers = config["pressure_layers"]
    deficit_fraction = float(config["reported_equilibrium_peak_deficit_fraction"])

    per_re: dict[str, dict] = {}
    hashes: dict[str, str] = {}
    for re_tau in config["training_re_tau"]:
        data, source = load_budget(int(re_tau))
        hashes[str(source.relative_to(ROOT))] = sha256(source)
        reconstructed_balance = (
            data["production"]
            + data["turb_transport"]
            + data["visc_transport"]
            + data["pressure_strain"]
            + data["pressure_transport"]
            - data["epsilon"]
        )
        y = data["y_plus"]
        f_dns = data["production"] / data["epsilon"]
        buffer_mask = (y >= layers["buffer_primary"][0]) & (
            y <= layers["buffer_primary"][1]
        )
        buffer_indices = np.flatnonzero(buffer_mask)
        peak_index = int(buffer_indices[np.argmax(f_dns[buffer_mask])])
        y12_index = int(np.argmin(np.abs(y - 12.0)))
        per_re[str(re_tau)] = {
            "Re_tau_nominal": int(re_tau),
            "budget_sign_convention": (
                "P + T_turb + T_visc + Phi_p + Pi_p - epsilon = balance; "
                "epsilon is positive and the saved TKE pressure-strain trace "
                "Phi_p is analytically zero"
            ),
            "balance_max_abs_error_recomputed_minus_reported": float(
                np.max(np.abs(reconstructed_balance - data["reported_balance"]))
            ),
            "layers": {
                name: layer_metrics(data, bounds) for name, bounds in layers.items()
            },
            "deficit_attribution": {
                "buffer_DNS_peak": point_diagnostic(
                    data, peak_index, deficit_fraction
                ),
                "nearest_to_y_plus_12": point_diagnostic(
                    data, y12_index, deficit_fraction
                ),
            },
        }

    out = {
        "schema_version": "1.0",
        "script_sha256": sha256(Path(__file__)),
        "config_sha256": sha256(args.config),
        "source_sha256": hashes,
        "ratio_definition": "|Pi_p|/(|T_turb|+|T_visc|)",
        "counterfactual": {
            "F_without_pressure": "1-(T_turb+T_visc)/epsilon",
            "F_DNS_minus_F_without_pressure": "-Pi_p/epsilon",
            "interpretation": (
                "Signed plausibility diagnostic only. The equilibrium closure is "
                "coupled, so this is not a causal decomposition of its 35% deficit."
            ),
        },
        "per_Re_tau": per_re,
        "equal_profile_aggregate": aggregate_equal_profile(
            per_re, list(layers.keys())
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "profiles": len(per_re)}, indent=2))


if __name__ == "__main__":
    main()
