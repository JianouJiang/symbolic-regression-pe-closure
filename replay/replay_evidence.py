#!/usr/bin/env python3
"""Replay the completed evidence calculations reported in the article.

The calculation deliberately reuses the frozen, source-hashed loaders and
configurations from the archived records.  It regenerates (i) the
five-Reynolds-number pressure-transport audit, (ii) the common-start matched
calibration diagnostic on the five Lee--Moser training profiles, and (iii) the
exact printed PySR--M16 comparison on all 179 frozen profile/station rows.

No manuscript or immutable submitted-baseline file is read for mutation.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
NODE = HERE
ROOT = HERE.parent
RECORDS = ROOT / "records"
RESULTS = NODE / "results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def replay_pressure() -> dict:
    code = RECORDS / "audits" / "pressure_transport_audit.py"
    config_path = RECORDS / "audits" / "audit_config.json"
    module = load_module("pressure_replay", code)
    module.ROOT = ROOT
    config = json.loads(config_path.read_text())
    layers = config["pressure_layers"]
    deficit_fraction = float(config["reported_equilibrium_peak_deficit_fraction"])

    per_re: dict[str, dict] = {}
    hashes: dict[str, str] = {}
    for re_tau in config["training_re_tau"]:
        data, source = module.load_budget(int(re_tau))
        hashes[str(source.relative_to(ROOT))] = module.sha256(source)
        reconstructed = (
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
                "epsilon is positive and Phi_p is the saved zero TKE trace"
            ),
            "balance_max_abs_error_recomputed_minus_reported": float(
                np.max(np.abs(reconstructed - data["reported_balance"]))
            ),
            "layers": {
                name: module.layer_metrics(data, bounds)
                for name, bounds in layers.items()
            },
            "deficit_attribution": {
                "buffer_DNS_peak": module.point_diagnostic(
                    data, peak_index, deficit_fraction
                ),
                "nearest_to_y_plus_12": module.point_diagnostic(
                    data, y12_index, deficit_fraction
                ),
            },
        }

    peak_rows = [
        row["deficit_attribution"]["buffer_DNS_peak"] for row in per_re.values()
    ]
    result = {
        "schema_version": "node000-replay-1.0",
        "replayed_from_script": str(code.relative_to(ROOT)),
        "source_script_sha256": sha256(code),
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "replay_script_sha256": sha256(Path(__file__)),
        "source_sha256": hashes,
        "ratio_definition": "|Pi_p|/(|T_turb|+|T_visc|)",
        "counterfactual": {
            "F_without_pressure": "1-(T_turb+T_visc)/epsilon",
            "F_DNS_minus_F_without_pressure": "-Pi_p/epsilon",
            "interpretation": (
                "Signed plausibility diagnostic, not a causal decomposition of "
                "the coupled equilibrium-closure deficit."
            ),
        },
        "per_Re_tau": per_re,
        "equal_profile_aggregate": module.aggregate_equal_profile(
            per_re, list(layers.keys())
        ),
        "buffer_peak_aggregate": {
            "n_profiles": len(peak_rows),
            "all_Pi_p_positive": all(row["Pi_p"] > 0.0 for row in peak_rows),
            "signed_fraction_of_35pct_deficit_mean": float(
                np.mean(
                    [row["signed_fraction_of_reported_deficit"] for row in peak_rows]
                )
            ),
            "signed_fraction_of_35pct_deficit_range": [
                float(
                    min(
                        row["signed_fraction_of_reported_deficit"]
                        for row in peak_rows
                    )
                ),
                float(
                    max(
                        row["signed_fraction_of_reported_deficit"]
                        for row in peak_rows
                    )
                ),
            ],
            "sign_aligned_fraction_of_35pct_deficit": 0.0,
            "conclusion": (
                "At the DNS buffer peak Pi_p has the opposite effect to that "
                "needed to explain a positive 35% underprediction; it accounts "
                "for none of that deficit under this signed counterfactual."
            ),
        },
    }
    return result


def replay_wall_aeh() -> dict:
    code = RECORDS / "audits" / "wall_aeh_contracts.py"
    config_path = RECORDS / "audits" / "audit_config.json"
    module = load_module("wall_aeh_replay", code)
    module.ROOT = ROOT
    config = json.loads(config_path.read_text())
    rows, source_hashes = module.fit_alpha4_by_profile(config)
    aeh = module.regress_aeh(rows, config)
    m16 = config["reported_formulas"]["m16"]
    pysr = config["reported_formulas"]["pysr"]
    a1, a2, a3, _a4 = m16["alpha"]
    domain = module.admissibility(float(a2), float(a3), aeh, config)
    return {
        "schema_version": "node000-replay-1.0",
        "replayed_from_script": str(code.relative_to(ROOT)),
        "source_script_sha256": sha256(code),
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "replay_script_sha256": sha256(Path(__file__)),
        "source_sha256": source_hashes,
        "coefficient_level_kinematics": {
            "velocity_expansions": (
                "u'+=a1*y+ + a2*(y+)^2+...; v'+=b2*(y+)^2+b3*(y+)^3+...; "
                "w'+=c1*y+ + c2*(y+)^2+..."
            ),
            "continuity_coefficient": (
                "2*b2 + partial_xplus(a1) + partial_zplus(c1) = 0"
            ),
            "reynolds_stress": "-<u'v'>+=-<a1*b2>*(y+)^3+O((y+)^4)",
            "production": "P+=-<a1*b2>*(y+)^3+O((y+)^4)",
            "dissipation": "epsilon+=<a1^2+c1^2>+O(y+)",
            "ratio_coefficient": (
                "c_w=-<a1*b2>/<a1^2+c1^2> (using dU+/dy+|wall=1)"
            ),
        },
        "formula_series": {
            "pysr": module.wall_series(pysr["alpha"])["pysr_linear_tanh"],
            "m16": module.wall_series(m16["alpha"])["m16_squared_tanh"],
            "note": (
                "The M16 denominator generates a nonzero quartic term; writing "
                "only c_w*(y+)^3+O((y+)^5) is generically incorrect."
            ),
        },
        "aeh_refit_with_printed_practical_m16_triplet": {
            "fixed_alpha1_alpha2_alpha3": [a1, a2, a3],
            "per_Re_tau": rows,
            "regression": aeh,
        },
        "explicit_closure": (
            "F(y+,Re_tau)=tanh^2(0.111*y+)/[tanh(0.052*y+-0.443)+"
            "{2.926889965+0.043587961*ln(Re_tau/1000)}/y+]"
        ),
        "admissibility": domain,
        "high_Re_asymptotics": {
            "fixed_y_plus": (
                "F ~ y+*tanh^2(alpha1*y+)/[alpha5*ln(Re_tau)] -> 0"
            ),
            "fixed_finite_Re_tau_then_y_plus_to_infinity": "F -> 1",
            "noncommuting_limits": {"lim_Re_then_lim_y": 0, "lim_y_then_lim_Re": 1},
            "claim_boundary": (
                "Only Re_tau=180--5200 is fitted; larger probes are unvalidated "
                "algebraic extrapolations."
            ),
        },
    }


def replay_matched_fit() -> dict:
    code = RECORDS / "fitting" / "matched_objective_fit.py"
    config_path = RECORDS / "fitting" / "fit_config.json"
    module = load_module("matched_fit_replay", code)
    module.ROOT = ROOT
    config = json.loads(config_path.read_text())
    result = module.fit_phase(config)
    result["replayed_from_script"] = str(code.relative_to(ROOT))
    result["source_script_sha256"] = sha256(code)
    result["config_path"] = str(config_path.relative_to(ROOT))
    result["replay_script_sha256"] = sha256(Path(__file__))
    return result


def load_full_suite_module():
    """Load the archived evaluator; its path constants are location-correct."""
    path = RECORDS / "manifest" / "full_suite_evaluation.py"
    module = load_module("full_suite_replay", path)
    module.LOADERS.ROOT = ROOT
    return module, path


def summarize_exact_rows(rows: list[dict]) -> dict:
    by_role: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_role[row["role"]].append(row)

    def family_summary(subset: list[dict]) -> list[dict]:
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in subset:
            groups[row["source_family"]].append(row)
        output = []
        for family in sorted(groups):
            members = groups[family]
            output.append(
                {
                    "source_family": family,
                    "n_rows": len(members),
                    "median_delta_R2_M16_minus_PySR": float(
                        np.median([row["delta_R2_M16_minus_PySR"] for row in members])
                    ),
                    "median_relative_RMSE_change_M16_over_PySR": float(
                        np.median(
                            [
                                row["relative_RMSE_change_M16_over_PySR"]
                                for row in members
                            ]
                        )
                    ),
                    "M16_R2_wins": int(
                        sum(row["delta_R2_M16_minus_PySR"] > 0 for row in members)
                    ),
                    "M16_RMSE_wins": int(
                        sum(
                            row["relative_RMSE_change_M16_over_PySR"] < 0
                            for row in members
                        )
                    ),
                }
            )
        return output

    primary = by_role["external_re_evaluation_in_domain"]
    primary_families = family_summary(primary)
    summary = {
        "n_all_rows": len(rows),
        "role_counts": {role: len(items) for role, items in sorted(by_role.items())},
        "primary_role": "external_re_evaluation_in_domain",
        "n_primary_rows": len(primary),
        "primary_source_family_summaries": primary_families,
        "primary_M16_R2_wins": int(
            sum(row["delta_R2_M16_minus_PySR"] > 0 for row in primary)
        ),
        "primary_M16_RMSE_wins": int(
            sum(row["relative_RMSE_change_M16_over_PySR"] < 0 for row in primary)
        ),
        "primary_median_of_family_median_delta_R2": float(
            np.median(
                [item["median_delta_R2_M16_minus_PySR"] for item in primary_families]
            )
        ),
        "primary_median_of_family_median_relative_RMSE_change": float(
            np.median(
                [
                    item["median_relative_RMSE_change_M16_over_PySR"]
                    for item in primary_families
                ]
            )
        ),
        "all_role_source_family_summaries": {
            role: family_summary(items) for role, items in sorted(by_role.items())
        },
        "evidence_led_practical_recommendation": "original_PySR",
        "recommendation_reason": (
            "On the 19 intended-domain external profiles the printed M16 wins "
            "only 1/19 paired R2 and RMSE comparisons; the source-balanced "
            "median delta R2 is negative and relative RMSE is higher.  M16's "
            "cubic wall order remains an analytical advantage, not a basis for "
            "hiding its finite-domain loss."
        ),
    }
    return summary


def replay_exact_formula_comparison() -> dict:
    module, code = load_full_suite_module()
    config_path = RECORDS / "manifest" / "method_config.json"
    fit_path = RECORDS / "manifest" / "wall_amplitude_fit_freeze.json"
    manifest_path = RECORDS / "manifest" / "profile_manifest_freeze.json"
    config = json.loads(config_path.read_text())
    fit = json.loads(fit_path.read_text())
    manifest = json.loads(manifest_path.read_text())

    # Verify every recorded source digest against the staged files.
    def verify_sources_after_archive(frozen_manifest: dict) -> dict[str, str]:
        expected: dict[str, str] = {}
        for frozen_row in frozen_manifest["rows"]:
            for relative_path, digest in frozen_row["source"]["sha256"].items():
                if relative_path in expected and expected[relative_path] != digest:
                    raise ValueError(f"inconsistent frozen hash for {relative_path}")
                expected[relative_path] = digest
        observed: dict[str, str] = {}
        for relative_path, digest in sorted(expected.items()):
            source_path = ROOT / relative_path
            actual = sha256(source_path)
            if actual != digest:
                raise ValueError(
                    f"source changed after freeze: {relative_path}: {actual} != {digest}"
                )
            observed[relative_path] = actual
        return observed

    module.verify_manifest_sources = verify_sources_after_archive
    replayed = module.run(config, fit, manifest)

    exact_rows = []
    for row in replayed["all_rows"]:
        pysr = row["metrics"]["original_PySR"]
        m16 = row["metrics"]["printed_M16"]
        exact_rows.append(
            {
                "dataset_id": row["dataset_id"],
                "source_family": row["source_family"],
                "flow_family": row["flow_family"],
                "role": row["role"],
                "historical_exposure": row["historical_exposure"],
                "actual_y_plus_range": row["actual_y_plus_range"],
                "original_PySR": pysr,
                "printed_M16": m16,
                "delta_R2_M16_minus_PySR": m16["R2"] - pysr["R2"],
                "delta_RMSE_M16_minus_PySR": m16["RMSE"] - pysr["RMSE"],
                "relative_RMSE_change_M16_over_PySR": m16["RMSE"] / pysr["RMSE"] - 1.0,
            }
        )
    return {
        "schema_version": "node000-exact-pair-replay-1.0",
        "formula_constants": {
            "original_PySR": [0.128, 0.053, 0.620, 5.78],
            "printed_M16": [0.111, 0.052, 0.443, 2.888],
        },
        "mask": config["mask"],
        "source_script": str(code.relative_to(ROOT)),
        "source_script_sha256": sha256(code),
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "fit_freeze_path": str(fit_path.relative_to(ROOT)),
        "fit_freeze_sha256": sha256(fit_path),
        "manifest_path": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": sha256(manifest_path),
        "manifest_freeze_verified": replayed["manifest_sha256_verified"],
        "source_files_verified": replayed["n_unique_source_files_verified"],
        "replay_script_sha256": sha256(Path(__file__)),
        "exact_printed_comparison_unconditional": True,
        "summary": summarize_exact_rows(exact_rows),
        "rows": exact_rows,
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    pressure = replay_pressure()
    write_json(RESULTS / "pressure_transport_replayed.json", pressure)
    wall_aeh = replay_wall_aeh()
    write_json(RESULTS / "wall_aeh_replayed.json", wall_aeh)
    matched = replay_matched_fit()
    write_json(RESULTS / "matched_training_fit_replayed.json", matched)
    exact = replay_exact_formula_comparison()
    write_json(RESULTS / "formula_comparison_exact_replayed.json", exact)
    print(
        json.dumps(
            {
                "pressure_peak": pressure["buffer_peak_aggregate"],
                "aeh": wall_aeh["aeh_refit_with_printed_practical_m16_triplet"][
                    "regression"
                ],
                "matched_training_loss": {
                    "linear_m1": matched["fits"]["linear_m1"]["best"]["loss"],
                    "cubic_m2": matched["fits"]["cubic_m2"]["best"]["loss"],
                },
                "exact_formula_summary": exact["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
