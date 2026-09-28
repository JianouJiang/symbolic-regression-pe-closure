# Changelog

## Unreleased — corrections to the published article (28 September 2026)

- Document three corrections to the published article in the README
  ("Corrections to the published article"): the plane Couette budgets were read
  with the wrong column, and the closure also describes that flow (R² 0.97–0.99);
  the 88–95 % dissipation result holds over 5 < y+ < 80; and Fig. 3(d) showed
  zero-pressure-gradient profiles instead of the adverse-pressure-gradient series.
- Add `corrections/` with scripts that reproduce the published and the corrected
  numbers, and the corrected Fig. 3, which the README now shows.

The closure, its coefficients and the evidence records are unchanged.

## 1.0.0 — 2026-09-17

First versioned release of the published Physics of Fluids article's code and
reproducibility records.

- Add a five-minute notebook, a Google Colab entry point, and a command-line
  example comparing the recommended PySR closure, the M16 benchmark, and the
  equilibrium production–dissipation assumption.
- Download a small DNS table from the original Lee–Moser provider, verify its
  checksum, and reproduce the published nominal-550 dissipation diagnostic.
- Document the data citation, identical evaluation masks, training-family
  status, offline use, and the distinction from coupled RANS validation.
- Add software-archive metadata while preserving the journal article as the
  preferred citation.

The published closure coefficients and original numerical evidence records
are unchanged.
