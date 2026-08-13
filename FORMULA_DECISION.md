# Formula decision for the Physics of Fluids revision

## Decision

The **original printed PySR expression** is the single formula recommended for practical finite-domain prediction in the revised manuscript:

\[
F_{\mathrm{PySR}}(y^+)=
\frac{\tanh(0.128y^+)}
{\tanh(0.053y^+-0.620)+5.78/y^+}.
\]

Its stated domain remains
\(1<y^+\leq\min(150,0.3Re_\tau)\) for incompressible, smooth-wall, pressure-driven flows, with the adverse-pressure-gradient and Couette limitations stated explicitly.

The printed M16 expression is retained only as a **physics-constrained asymptotic benchmark**. It enforces the exact cubic wall exponent, whereas the recommended PySR expression has a quadratic leading wall order. M16 is not presented as the recommended predictive closure because its finite-domain comparison is unfavourable.

## Evidence

Both printed formulas were evaluated on the same materialised profiles, point masks, and source-hashed 179-row manifest. The decision population was the 19 intended-domain external profiles from four independent source families; APG, separated-flow, and Couette cases were reported as stress tests but did not vote in the recommendation.

- M16 won only 1 of 19 paired comparisons in both \(R^2\) and RMSE.
- The median of source-family median \(\Delta R^2=R^2_{\rm M16}-R^2_{\rm PySR}\) was \(-0.0043843\).
- The median of source-family median relative RMSE change was \(+17.759\%\), so M16 was worse on the primary decision statistic.
- M16 lost all paired comparisons in the UPM channel, KTH ZPG boundary-layer, and El Khoury pipe families; it won the single Yao high-Reynolds-number pipe profile.

The exact cubic exponent is therefore reported as M16's analytical advantage, not used to conceal its finite-domain predictive loss. This comparison adjudicates the two submitted printed coefficient sets; it does not establish that every possible cubic recalibration is inferior. Selecting another calibration on source families already exposed during historical model selection would require a prospectively frozen study and is not part of this minor revision.

## Reproducible records

- `replay/results/formula_comparison_exact_replayed.json`
  - SHA-256: `d25ab68a4b64840aad9c41ca210c3ceeaa9ce00dccdb09d3ef0bba75eb50b5c0`
- `replay/results/pressure_transport_replayed.json`
  - SHA-256: `7e54dfc0360ae092fc99412b28013877a23621676c7b4e37eb2b250a2c17fffa`
- `replay/results/wall_aeh_replayed.json`
  - SHA-256: `e93959e2f439e30059e56f5e9acb547129b4f8231325ebf3d2c288bb65c13dc3`
- `replay/replay_evidence.py`
  - SHA-256: `fe028a555661dd5bc0fc078561a8e79809262cffe2a14419dbce8e2cafa2e233`

## Excluded revision-only exploration

The later hard-anchored and dual-activation candidates were generated after submission while exploring a more ambitious follow-up. They are not necessary to answer the referees, were not part of the submitted formula pair, and are excluded from the manuscript, response, figures, conclusions, and submission package for this minor revision.
