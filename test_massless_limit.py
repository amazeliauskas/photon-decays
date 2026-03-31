#!/usr/bin/env python
# coding: utf-8
"""
Test that the massive result converges smoothly to the massless result as mb=mc -> 0.
Tests both pT < ma/2 (am < -1 branch needed) and pT > ma/2 (am >= -1 only).
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Import functions from Photon_decay (avoid running the main script)
import importlib.util, types

# Manually import only the functions we need
import numpy as np
import scipy.integrate as integrate
from scipy.optimize import brentq
from scipy.special import ellipk

exec(open("Photon_decay.py").read().split("# pi^0 ->")[0])  # load up to main script

ma = 0.14
T  = 0.230

def thermaldNadpT(p, m, T):
    E = np.sqrt(p**2 + m**2)
    return p / (np.exp(E/T) - 1)

# pT values: below and above ma/2 = 0.07
pT_test = np.array([0.01, 0.03, 0.05, 0.06, 0.10, 0.20, 0.50, 1.00])

# Reference: massless result
mb0, mc0 = 0.0, 0.0
ref, _ = zip(*[getFeedDown_anadNa(np.array([p]), ma, mb0, mc0, 1,
                                   thermaldNadpT, (ma, T)) for p in pT_test])
ref = np.array([v[0] for v in ref])

print(f"ma = {ma},  T = {T},  ma/2 = {ma/2:.4f}")
print()
print(f"{'pT':>6}  {'massless':>14}", end="")
for mb in [0.01, 0.005, 0.001]:
    print(f"  {'mb='+str(mb):>12}  {'rel.diff':>10}", end="")
print()
print("-" * (6 + 14 + 3*(12+10+4)))

for j, p in enumerate(pT_test):
    branch = "am<-1" if p < ma/2 else "am>=-1"
    print(f"{p:6.3f}  {ref[j]:14.8f}", end="")
    for mb in [0.01, 0.005, 0.001]:
        vals, _ = getFeedDown_anadNa(np.array([p]), ma, mb, mb, 1,
                                      thermaldNadpT, (ma, T))
        v = vals[0]
        rel = (v - ref[j]) / ref[j] if ref[j] != 0 else float('nan')
        print(f"  {v:12.8f}  {rel:+10.4e}", end="")
    print(f"  [{branch}]")
