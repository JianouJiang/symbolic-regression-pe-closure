"""Compact inner-layer production-dissipation closure.

Implements the two printed formulas of

    Jiang & Rosic, "Combining Symbolic Regression and Turbulence Physics to
    Develop a Compact Inner-Layer Production--Dissipation Closure",
    Physics of Fluids (accepted).

F(y+) approximates the turbulent-kinetic-energy production-to-dissipation
ratio P/eps as a function of the inner-scaled wall distance y+ on the domain

    1 < y+ <= min(150, 0.3 * Re_tau)

for incompressible, smooth-wall, pressure-driven flows.  The PySR form is the
recommended predictive closure; the M16 form is a physics-constrained
asymptotic benchmark that enforces the kinematic (y+)^3 wall exponent.
Accuracy deteriorates for strong adverse pressure gradients, and the closure
is not an outer-wake model.
"""
import numpy as np

# Printed coefficient sets (article Eq. for the paired family, m = 1 and m = 2).
ALPHA_PYSR = (0.128, 0.053, 0.620, 5.78)   # m = 1, recommended closure
ALPHA_M16 = (0.111, 0.052, 0.443, 2.888)   # m = 2, asymptotic benchmark


def _family(yplus, alpha, m):
    yp = np.asarray(yplus, dtype=float)
    a1, a2, a3, a4 = alpha
    return np.tanh(a1 * yp) ** m / (np.tanh(a2 * yp - a3) + a4 / yp)


def f_pysr(yplus):
    """Recommended closure F_PySR(y+) = tanh(0.128 y+) / (tanh(0.053 y+ - 0.620) + 5.78/y+)."""
    return _family(yplus, ALPHA_PYSR, m=1)


def f_m16(yplus):
    """M16 asymptotic benchmark (squared numerator; exact cubic wall exponent)."""
    return _family(yplus, ALPHA_M16, m=2)


def valid_domain(yplus, re_tau):
    """Boolean mask for the stated domain 1 < y+ <= min(150, 0.3 Re_tau)."""
    yp = np.asarray(yplus, dtype=float)
    return (yp > 1.0) & (yp <= min(150.0, 0.3 * float(re_tau)))


if __name__ == "__main__":
    yp = np.array([2.0, 5.0, 12.0, 30.0, 60.0, 100.0, 150.0])
    print("  y+      F_PySR     F_M16")
    for y, a, b in zip(yp, f_pysr(yp), f_m16(yp)):
        print(f"{y:6.1f}  {a:9.4f} {b:9.4f}")
    print("\nDomain mask at Re_tau = 550:", valid_domain(yp, 550.0))
