#!/usr/bin/env python3
"""Run the published closure on one checksum-verified Lee--Moser DNS profile."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import sys
from urllib.request import Request, urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from closure.pe_closure import f_m16, f_pysr, valid_domain

SOURCE_URL = (
    "https://turbulence.oden.utexas.edu/channel2015/data/"
    "LM_Channel_0550_RSTE_k_prof.dat"
)
SOURCE_SHA256 = "1b8d65696b6647b9131cd7b033fb82076bcb831c72280d1c932c3c63d3ee0068"
SOURCE_CITATION = (
    "M. Lee and R. D. Moser, Journal of Fluid Mechanics 774, 395--415 (2015). "
    "doi:10.1017/jfm.2015.268"
)
DEFAULT_CACHE = ROOT / "examples" / ".cache" / "LM_Channel_0550_RSTE_k_prof.dat"


def verified_bytes(payload: bytes) -> bytes:
    """Fail explicitly if the upstream table has changed or a cache is corrupt."""
    actual = hashlib.sha256(payload).hexdigest()
    if actual != SOURCE_SHA256:
        raise ValueError(
            f"DNS checksum mismatch: expected {SOURCE_SHA256}, received {actual}. "
            "Check the original provider's revision notes before using a changed table."
        )
    return payload


def load_dns(data_path: Path | None = None, cache_path: Path = DEFAULT_CACHE) -> dict:
    """Download about 47 kB once, or use a verified local copy without network access."""
    path = Path(data_path) if data_path is not None else Path(cache_path)
    if path.is_file():
        payload = verified_bytes(path.read_bytes())
    elif data_path is not None:
        raise FileNotFoundError(f"Requested DNS table does not exist: {path}")
    else:
        request = Request(SOURCE_URL, headers={"User-Agent": "pe-closure-tutorial/1.0"})
        try:
            with urlopen(request, timeout=30) as response:
                payload = verified_bytes(response.read())
        except OSError as exc:
            raise RuntimeError(
                f"Could not download {SOURCE_URL}. Retry with internet access, or "
                "supply a previously downloaded original table with --data PATH."
            ) from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    text = payload.decode("utf-8")
    match = re.search(r"Re_tau\s+Re_tau\s*=\s*([\d.]+)", text)
    if match is None:
        raise ValueError("The source table is missing its friction Reynolds number.")
    raw = np.loadtxt(io.StringIO(text), comments="%")
    if raw.ndim != 2 or raw.shape[1] != 9 or not np.all(np.isfinite(raw)):
        raise ValueError("Expected a finite, nine-column Lee--Moser energy-budget table.")
    # Zero-based columns: 1 = y+, 2 = production, 7 = positive dissipation.
    return {
        "yplus": raw[:, 1],
        "production": raw[:, 2],
        "epsilon": raw[:, 7],
        "re_tau": float(match.group(1)),
    }


def error_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict:
    residual = prediction - reference
    variance = np.sum((reference - np.mean(reference)) ** 2)
    return {
        "RMSE": float(np.sqrt(np.mean(residual ** 2))),
        "R2": float(1.0 - np.sum(residual ** 2) / variance),
    }


def evaluate(dns: dict) -> dict:
    y, production, epsilon = (dns[key] for key in ("yplus", "production", "epsilon"))
    domain = valid_domain(y, dns["re_tau"]) & (epsilon > 1e-8)
    diagnostic = domain & (y > 5.0) & (y < 80.0)
    ratio_rows, epsilon_rows = {}, {}
    # Evaluate only positive, in-domain y+: the formulas are singular at y+ = 0.
    for name, formula in (
        ("Equilibrium", np.ones_like),
        ("PySR (recommended)", f_pysr),
        ("M16 (asymptotic benchmark)", f_m16),
    ):
        ratio_rows[name] = error_metrics(production[domain] / epsilon[domain], formula(y[domain]))
        epsilon_rows[name] = error_metrics(epsilon[diagnostic], production[diagnostic] / formula(y[diagnostic]))
    baseline = epsilon_rows["Equilibrium"]["RMSE"]
    for row in epsilon_rows.values():
        row["RMSE_reduction_vs_equilibrium_percent"] = 100.0 * (1.0 - row["RMSE"] / baseline)
    return {
        "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "source_citation": SOURCE_CITATION,
        "article_doi": "10.1063/5.0347368",
        "actual_re_tau": dns["re_tau"],
        "dataset_role": "Training-family illustration; not a new independent validation.",
        "ratio": {
            "mask": "1 < y+ <= min(150, 0.3 Re_tau), epsilon+ > 1e-8",
            "n": int(domain.sum()),
            "actual_yplus_range": [float(y[domain].min()), float(y[domain].max())],
            "models": ratio_rows,
        },
        "dissipation_diagnostic": {
            "mask": "5 < y+ < 80, within the closure domain, epsilon+ > 1e-8",
            "n": int(diagnostic.sum()),
            "actual_yplus_range": [float(y[diagnostic].min()), float(y[diagnostic].max())],
            "models": epsilon_rows,
            "interpretation": "A-priori diagnostic using DNS production; not a coupled RANS result.",
        },
    }


def make_figure(dns: dict):
    import matplotlib.pyplot as plt

    y, production, epsilon = (dns[key] for key in ("yplus", "production", "epsilon"))
    domain = valid_domain(y, dns["re_tau"]) & (epsilon > 1e-8)
    diagnostic = domain & (y > 5.0) & (y < 80.0)
    styles = [("PySR (recommended)", f_pysr, "#1769aa"),
              ("M16 (benchmark)", f_m16, "#d17818")]
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
        grid = np.geomspace(1.0001, min(150, 0.3 * dns["re_tau"]), 400)
        axes[0].semilogx(y[domain], production[domain] / epsilon[domain], ".", color="#263238", ms=5, label="DNS")
        for name, formula, color in styles:
            axes[0].semilogx(grid, formula(grid), color=color, lw=2, label=name)
        axes[0].axhline(1, color="#616161", ls="--", lw=1.5, label="Equilibrium")
        axes[0].set(xlabel=r"Wall distance $y^+$", ylabel=r"$\mathcal{P}/\varepsilon$",
                    title="The ratio varies across the inner layer", xlim=(1, 150))
        axes[0].legend(frameon=False, fontsize=9)

        yd, pd, ed = y[diagnostic], production[diagnostic], epsilon[diagnostic]
        axes[1].plot(yd, ed, ".", color="#263238", ms=6, label="DNS dissipation")
        axes[1].plot(yd, pd, color="#616161", ls="--", lw=1.5, label=r"Equilibrium: $\varepsilon^+=\mathcal{P}^+$")
        for name, formula, color in styles:
            axes[1].plot(yd, pd / formula(yd), color=color, lw=2, label=name)
        axes[1].set(xlabel=r"Wall distance $y^+$", ylabel=r"Dissipation $\varepsilon^+$",
                    title="Using DNS production to estimate dissipation", xlim=(5, 80))
        axes[1].legend(frameon=False, fontsize=9)
        for ax in axes:
            ax.grid(alpha=0.15)
        fig.suptitle(r"Lee–Moser channel DNS: $Re_\tau=543.496$ (nominal 550), no refitting", fontsize=13)
        fig.text(0.5, 0.025, "Training-family example · DNS-based diagnostic · No coupled RANS performance claim", ha="center", fontsize=10, color="#455a64")
        fig.tight_layout(rect=(0, 0.065, 1, 0.93))
    return fig


def print_metrics(results: dict) -> None:
    print(f"Lee–Moser channel: Re_tau = {results['actual_re_tau']}")
    print(results["dataset_role"])
    for key, title in [("ratio", "Production/dissipation ratio"), ("dissipation_diagnostic", "Dissipation using DNS production")]:
        block = results[key]
        print(f"\n{title}: {block['mask']} ({block['n']} points)")
        print(f"{'Model':30s} {'RMSE':>12s} {'R2':>12s}")
        for name, row in block["models"].items():
            print(f"{name:30s} {row['RMSE']:12.8f} {row['R2']:12.8f}")
    reduction = results["dissipation_diagnostic"]["models"]["PySR (recommended)"]["RMSE_reduction_vs_equilibrium_percent"]
    print(f"\nPySR dissipation RMSE reduction vs equilibrium: {reduction:.2f}%")
    print(results["dissipation_diagnostic"]["interpretation"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Use a local original DNS table (checksum verified).")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "examples" / "output")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    dns = load_dns(args.data)
    results = evaluate(dns)
    print_metrics(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    figure = make_figure(dns)
    figure.savefig(args.output_dir / "quickstart.png", dpi=160)
    figure.savefig(args.output_dir / "quickstart.pdf")
    print(f"\nFigure and metrics: {args.output_dir}")


if __name__ == "__main__":
    main()
