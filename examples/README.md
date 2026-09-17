# Five-minute tutorial

Run the published production–dissipation closure on a small channel-flow DNS example.
No simulation, model fitting, PySR installation, or manual dataset staging is needed.

[Open the notebook](five_minute_tutorial.ipynb) ·
[Run in Google Colab](https://colab.research.google.com/github/JianouJiang/symbolic-regression-pe-closure/blob/v1.0.0/examples/five_minute_tutorial.ipynb)

In Colab, choose **Runtime → Run all**. The notebook fetches the versioned public
code and one 47 kB DNS table. Locally, from the repository root:

```bash
python3 -m pip install -r requirements.txt
python3 examples/quickstart.py
```

The script saves `examples/output/quickstart.png`, `quickstart.pdf`, and
`metrics.json`. To use the notebook locally, install JupyterLab and open
`examples/five_minute_tutorial.ipynb` from the cloned repository.

## What you will see

1. The DNS production-to-dissipation ratio, the recommended PySR expression,
   the M16 asymptotic benchmark, and the equilibrium assumption `P/epsilon = 1`.
2. An a-priori dissipation estimate `epsilon = P_DNS / F(y+)` compared with
   DNS dissipation and `epsilon = P_DNS`.
3. RMSE and R² computed on identical native DNS points for every model.

The ratio uses `1 < y+ <= min(150, 0.3 Re_tau)`. The dissipation diagnostic
uses `5 < y+ < 80` within that domain. At nominal `Re_tau = 550`, the
recommended expression reduces dissipation RMSE by **94.61%** on those 49
points, reproducing the corresponding entry in the paper's
[existing numerical audit](../audit/CLEAN_ROOM_NUMERICAL_AUDIT.json).

This Lee–Moser profile belongs to the **training dataset family**. It is an
illustration of how to use the fixed formulas, not additional independent
validation. Production is taken from DNS; this example does not demonstrate
performance in a coupled RANS simulation. Strong adverse pressure gradients,
outer-wake prediction, and plane Couette flow are outside the demonstrated
scope. The PySR expression is not the exact cubic wall-limit expression;
M16 is retained for that asymptotic comparison.

## Data provenance and offline use

The tutorial downloads the original
[Lee–Moser energy-budget table](https://turbulence.oden.utexas.edu/channel2015/data/LM_Channel_0550_RSTE_k_prof.dat)
and verifies SHA-256
`1b8d65696b6647b9131cd7b033fb82076bcb831c72280d1c932c3c63d3ee0068`.
The header gives the actual `Re_tau = 543.496`; 550 is the dataset label.
Columns 2, 3, and 8 (one-based) are `y+`, production, and **positive**
viscous dissipation in wall units. No interpolation or fitting is performed.

The downloaded table is cached in `examples/.cache/`, which is excluded
from Git and the release. Data remain subject to their original provider's
terms and are not relicensed under this repository's MIT licence.
The [provider requests citation](https://turbulence.oden.utexas.edu/channel2015/content/README_2015.html)
of **M. Lee and R. D. Moser, “Direct numerical simulation of turbulent channel
flow up to Re_tau = 5200,” Journal of Fluid Mechanics 774, 395–415 (2015),
[doi:10.1017/jfm.2015.268](https://doi.org/10.1017/jfm.2015.268)**.

After the first successful download, subsequent runs use the verified cache.
You can also supply the original file yourself:

```bash
python3 examples/quickstart.py --data /path/to/LM_Channel_0550_RSTE_k_prof.dat
```

A download failure or checksum mismatch stops the example with an explanatory
error. It never substitutes synthetic or unverified data.

## Cite the closure

J. Jiang and B. Rosic, “Combining symbolic regression and turbulence physics
to develop a compact inner-layer production–dissipation closure,”
*Physics of Fluids* **38**, 095140 (2026).
[doi:10.1063/5.0347368](https://doi.org/10.1063/5.0347368).
