#!/usr/bin/env python
# coding: utf-8
"""
Numerical test of the formula for the phi-integral in the a_- < -1 case:

  integral_{-1}^{1} dx / sqrt((1-x^2)(x - a_-)(x + a_+))
    = 2 K(k^2) / sqrt((1 - a_-)(a_+ - 1))

with  k^2 = 2(a_- + a_+) / ((1 - a_-)(a_+ - 1)),  0 < k^2 < 1.

Requirements: a_- < -1,  a_+ > 1,  a_- + a_+ > 0  (so -a_+ < a_- < -1 < 1).
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import ellipk


def integral_numerical(am, ap):
    """Direct numerical integration of the elliptic integral from -1 to 1."""
    def integrand(x):
        return 1.0 / np.sqrt((1 - x**2) * (x - am) * (x + ap))
    val, err = quad(integrand, -1, 1, limit=200, epsabs=1e-12, epsrel=1e-10)
    return val, err


def integral_formula(am, ap):
    """Closed-form result via complete EllipticK."""
    k2 = 2*(am + ap) / ((1 - am)*(ap - 1))
    return 2*ellipk(k2) / np.sqrt((1 - am)*(ap - 1))


# Test cases: (a_-, a_+) with a_- < -1, a_+ > 1, a_- + a_+ >= 0
# (a_- + a_+ > 0 is always guaranteed physically; = 0 is the degenerate k^2 = 0 case)
test_cases = [
    (-1.5,  2.0),   # generic
    (-2.0,  2.5),   # generic
    (-2.0,  3.0),
    (-3.0,  4.0),
    (-1.1,  1.5),   # close to boundary a_- = -1 → k^2 close to 1
    (-5.0, 10.0),   # large values
    (-1.01, 1.5),   # very close to a_- = -1, k^2 -> 1 (log-divergent K but finite integral)
    (-2.0,  2.0),   # degenerate: a_- + a_+ = 0 → k^2 = 0, K(0) = pi/2
]

print(f"{'am':>8} {'ap':>8} {'k^2':>10} {'numerical':>16} {'formula':>16} {'rel. error':>12}")
print("-" * 76)

all_passed = True
for am, ap in test_cases:
    assert am < -1 and ap > 1 and am + ap > 0, f"Preconditions failed for ({am}, {ap})"
    k2 = 2*(am + ap) / ((1 - am)*(ap - 1))
    num, num_err = integral_numerical(am, ap)
    form = integral_formula(am, ap)
    rel_err = abs(num - form) / abs(form)
    passed = rel_err < 1e-6
    all_passed = all_passed and passed
    flag = "" if passed else "  FAIL"
    print(f"{am:8.3f} {ap:8.3f} {k2:10.6f} {num:16.10f} {form:16.10f} {rel_err:12.2e}{flag}")

print()
print("All tests passed." if all_passed else "SOME TESTS FAILED.")
