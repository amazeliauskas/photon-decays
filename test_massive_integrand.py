#!/usr/bin/env python
# coding: utf-8
"""
Test the phi-integral formula in integrand_factor_massive for both branches.

The phi-integral (after eliminating the rapidity delta-function) is:

  I_phi = integral_{-phi_+}^{phi_+} dphi  2 / sqrt((pT qT cos(phi) + ma E*)^2 - mqT^2 mpT^2)

where phi_+ = arccos(a_-) for a_- in [-1,1] and phi_+ = pi for a_- < -1.

After the substitution x = cos(phi) this equals  4/(pT qT) * mathcal{I}(a_-, a_+), where

  mathcal{I} = sqrt(2) * sqrt((a_-+1)/(a_-+a_+)) / sqrt(a_-+1) * K(m),  m = (1-a_-)(a_+-1)/(2(a_-+a_+))   [a_- >= -1]
  mathcal{I} = 2 K(k^2) / sqrt((1-a_-)(a_+-1)),                          k^2 = 2(a_-+a_+)/((1-a_-)(a_+-1))  [a_- < -1]

We test that the direct numerical phi-integral matches the formula.
Particle: rho -> pi + pi  (ma=0.7711, mb=mc=0.13957 GeV).
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import ellipk


# ---------- kinematics helpers (copied from Photon_decay.py) ----------

def getEstar(ma, mb, mc):
    return (ma**2 + mb**2 - mc**2) / (2*ma)

def getmT(m, pT):
    return np.sqrt(m**2 + pT**2)

def getPstar(ma, mb, mc):
    term1 = (mc**2 - ma**2 - mb**2)**2
    return np.sqrt(term1 - 4*ma**2*mb**2) / (2*ma)

def getAminus(qT, ma, mb, mc, pT):
    Estar = getEstar(ma, mb, mc)
    return (-ma*Estar + getmT(ma, qT)*getmT(mb, pT)) / (pT*qT)

def getAplus(qT, ma, mb, mc, pT):
    Estar = getEstar(ma, mb, mc)
    return ( ma*Estar + getmT(ma, qT)*getmT(mb, pT)) / (pT*qT)


# ---------- phi-integral: direct numerical ----------

def phi_integral_numerical(qT, ma, mb, mc, pT):
    """Direct integration over azimuthal angle phi."""
    Estar = getEstar(ma, mb, mc)
    mqT   = getmT(ma, qT)
    mpT   = getmT(mb, pT)
    am    = (-ma*Estar + mqT*mpT) / (pT*qT)

    def integrand(phi):
        arg = (pT*qT*np.cos(phi) + ma*Estar)**2 - mqT**2*mpT**2
        return 2.0 / np.sqrt(arg) if arg > 0 else 0.0

    phi_plus = np.arccos(np.clip(am, -1.0, 1.0)) if am >= -1 else np.pi
    val, err = quad(integrand, -phi_plus, phi_plus,
                    limit=300, epsabs=1e-12, epsrel=1e-10)
    return val, err


# ---------- phi-integral: EllipticK formula ----------

def phi_integral_formula(qT, ma, mb, mc, pT):
    """Formula: 4/(pT qT) * mathcal{I}(a_-, a_+)."""
    am = getAminus(qT, ma, mb, mc, pT)
    ap = getAplus( qT, ma, mb, mc, pT)
    if am < -1:
        k2 = 2*(am + ap) / ((1 - am)*(ap - 1))
        phiI = 2*ellipk(k2) / np.sqrt((1 - am)*(ap - 1))
    else:
        m = (1 - am)*(ap - 1) / (2*(am + ap))
        phiI = np.sqrt(2)*np.sqrt((am + 1)/(am + ap)) / np.sqrt(am + 1) * ellipk(m)
    return 4/(pT*qT) * phiI


# ---------- run tests ----------

ma, mb, mc = 0.7711, 0.13957, 0.13957   # rho -> pi pi
pstar = getPstar(ma, mb, mc)
print(f"rho -> pi pi:  ma={ma}, mb=mc={mb},  p* = {pstar:.4f} GeV")
print()

# Build test cases: (pT, qT, expected branch)
#  pT > p*  → qTminus > 0  → a_- in [-1,1] throughout  (am >= -1 branch)
#  pT < p*  → qTminus < 0  → small qT gives a_- << -1   (am < -1 branch)
test_cases = [
    # --- am >= -1 branch (pT = 0.5 > p* ≈ 0.36) ---
    (0.5, 0.30, "am >= -1"),
    (0.5, 0.50, "am >= -1"),
    (0.5, 1.00, "am >= -1"),
    (0.5, 2.00, "am >= -1"),
    # --- am < -1 branch (pT = 0.1 < p*) ---
    (0.1, 0.05, "am < -1"),
    (0.1, 0.10, "am < -1"),
    (0.1, 0.20, "am < -1"),
    (0.1, 0.50, "am < -1"),
    # --- near the transition a_- = -1 (pT = 0.1, qT* ≈ 0.9) ---
    (0.1, 0.80, "am < -1"),
    (0.1, 1.00, "am >= -1"),
]

print(f"{'pT':>6} {'qT':>6} {'branch':>10} {'am':>10} {'numerical':>16} {'formula':>16} {'rel.err':>10}")
print("-" * 82)

all_passed = True
for pT, qT, branch_label in test_cases:
    am = getAminus(qT, ma, mb, mc, pT)
    num, num_err = phi_integral_numerical(qT, ma, mb, mc, pT)
    form = phi_integral_formula(qT, ma, mb, mc, pT)
    rel_err = abs(num - form) / abs(form)
    passed = rel_err < 1e-6
    all_passed = all_passed and passed
    actual_branch = "am < -1" if am < -1 else "am >= -1"
    branch_ok = "✓" if actual_branch == branch_label else "WRONG BRANCH"
    flag = "" if passed else "  FAIL"
    print(f"{pT:6.2f} {qT:6.2f} {actual_branch:>10}  {am:10.4f} {num:16.10f} {form:16.10f} {rel_err:10.2e}{flag}  {branch_ok}")

print()
print("All tests passed." if all_passed else "SOME TESTS FAILED.")
