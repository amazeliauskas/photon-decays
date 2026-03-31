#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy.integrate as integrate
from scipy.optimize import brentq
from scipy.special import ellipk
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.close('all')
mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


def getPstar(ma, mb, mc):
    try:
        term1 = (mc**2 - ma**2 - mb**2)**2
        term2 = 4*ma**2*mb**2
        if(term1 < term2):
            raise ValueError()
        elif ma==0:
            raise ZeroDivisionError()
        return np.sqrt(term1-term2)/(2*ma)
    except ValueError:
        print("Warning: Negative square root encountered in p*. ")
        return np.nan
    except ZeroDivisionError:
        print("Warning: Invalid value passed for parent particle, ma = 0")
        return np.nan

def getEstar(ma, mb, mc):
    try:
        return (ma**2 + mb**2 - mc**2)/(2*ma)
    except ZeroDivisionError:
        print("Warning: Invalid value passed for parent particle, ma = 0")
        return np.nan

def getmT(m, pT):
    return np.sqrt(m**2 + pT**2)

def get_qTplus(ma, mb, mc, pT):
    if mb != 0:
        pstar = getPstar(ma, mb, mc)
        mstarT = getmT(mb, pstar)
        mT = getmT(mb, pT)
        return ma/mb**2*(pT*mstarT + pstar*mT)
    else:
        return np.inf

def get_qTminus(ma, mb, mc, pT):
    pstar = getPstar(ma, mb, mc)
    mstarT = getmT(mb, pstar)
    mT = getmT(mb, pT)
    if (mb==0):
        if (mc==0):
            return 2*(pT**2 - ma**2/4)/(2*pT)
        else:
            return ma*(pT**2 - pstar**2)/(2*pT*pstar)
    else:
        return ma/mb**2 * (pT*mstarT - pstar*mT)

def getAplus(qT, ma, mb, mc, pT):
    Estar = getEstar(ma, mb, mc)
    mqT = getmT(ma, qT)
    mpT = getmT(mb, pT)
    return (ma*Estar + mqT*mpT)/pT/qT

def getAminus(qT, ma, mb, mc, pT):
    Estar = getEstar(ma, mb, mc)
    mqT = getmT(ma, qT)
    mpT = getmT(mb, pT)
    return (-ma*Estar + mqT*mpT)/pT/qT

def integrand_factor_massive(qT, b, ma, mb, mc, pT, Debug=False):
    try:
        pstar = getPstar(ma, mb, mc)
        norm = ma/pstar/np.pi*b
        am = getAminus(qT, ma, mb, mc, pT)
        ap = getAplus(qT, ma, mb, mc, pT)
        if (am < -1):
            # roots ordered -ap < am < -1 < 1; standard elliptic reduction (DLMF 19.7)
            # gives a complete EllipticK
            k2 = 2*(am+ap)/((1-am)*(ap-1))
            if not (0 < k2 < 1):
                if Debug:
                    print("Warning! k2 out of range in am<-1 branch. Returning NaN.")
                raise ValueError()
            phiIntegral = 2*ellipk(k2)/np.sqrt((1-am)*(ap-1))
        else:
            argument = (1 - am)*(ap-1)/(2*(am+ap))
            if (am+1) < 0 or not (0 <= argument < 1):
                if Debug:
                    print("Warning! Invalid EllipticK argument. Returning NaN.")
                raise ValueError()
            phiIntegral = np.sqrt(2)*np.sqrt((am + 1)/(am + ap))/np.sqrt(am+1)*ellipk(argument)
        return norm/qT*phiIntegral
    except ValueError:
        return 0.0  # outside physical range [qTminus, qTplus], kernel is zero
    except ZeroDivisionError:
        if Debug:
            print("Warning! Division by zero, returning 0")
        return 0.0

def integrand_factor_massless(qT, b, ma, mb, mc, pT, Debug=False):
    try:
        norm = 2*b/np.pi
        am = getAminus(qT, ma, mb, mc, pT)
        ap = getAplus(qT, ma, mb, mc, pT)
        if am < -1:
            # roots ordered -ap < am < -1 < 1; standard elliptic reduction (DLMF 19.7)
            # gives a complete EllipticK
            k2 = 2*(am+ap)/((1-am)*(ap-1))
            if not (0 < k2 < 1):
                if Debug:
                    print("Warning! k2 out of range in am<-1 branch. Returning NaN.")
                raise ValueError()
            phiIntegral = 2*ellipk(k2)/np.sqrt((1-am)*(ap-1))
        else:
            argument = (1 - am)*(ap-1)/(2*(am+ap))
            if (am+1) < 0 or not (0 <= argument < 1):
                if Debug:
                    print("Warning! Invalid EllipticK argument. Returning NaN.")
                raise ValueError()
            phiIntegral = np.sqrt(2)*np.sqrt((am+1)/(am + ap))/np.sqrt(am+1)*ellipk(argument)
        return norm*phiIntegral/qT
    except ValueError:
        return np.nan
    except ZeroDivisionError:
        if Debug:
            print("Warning! Division by zero, returning NaN")
        return np.nan


def get_qTstar(ma, mb, mc, pT):
    """Find qT* where a_- = -1, the transition between the two EllipticK branches.
    For massless (mb=mc=0) this is -qTminus exactly.
    For massive, solve getAminus(qT*) = -1 numerically."""
    if (mb == 0) and (mc == 0):
        return -get_qTminus(ma, mb, mc, pT)
    qTplus = get_qTplus(ma, mb, mc, pT)
    return brentq(lambda qT: getAminus(qT, ma, mb, mc, pT) + 1,
                  1e-10, qTplus, xtol=1e-12, rtol=1e-10)


def getFeedDown_anadNa(pT, ma, mb, mc, b, dNa_dpT, dNa_dpT_args, EpsRel=1e-4, Debug=False):
    vals = np.zeros_like(pT)
    vals_err = np.zeros_like(pT)
    for i, p in enumerate(pT):
        qTminus = get_qTminus(ma, mb, mc, p)
        qmin = max(qTminus, 0)
        qmax = get_qTplus(ma, mb, mc, p)
        args = (b, ma, mb, mc, p, Debug)
        if (mb == 0) and (mc == 0):
            integ = lambda qT: integrand_factor_massless(qT, *args)*dNa_dpT(qT, *dNa_dpT_args)
        else:
            integ = lambda qT: integrand_factor_massive(qT, *args)*dNa_dpT(qT, *dNa_dpT_args)
            # For massive case, qTplus ~ ma/mb^2 diverges as mb->0 causing quad to fail.
            # integrand_factor_massive already returns 0 outside [qTminus, qTplus], so
            # using np.inf lets quad use its efficient infinite-range substitution.
            qmax = np.inf
        if qTminus < 0:
            # Interior logarithmic singularity at qT* where am = -1 (EllipticK -> inf).
            # Split so quad sees it only as an endpoint singularity in each sub-interval.
            # Note: quad cannot use points= with infinite upper limits.
            qstar = get_qTstar(ma, mb, mc, p)
            v1, e1 = integrate.quad(integ, qmin, qstar, epsabs=1e-10, epsrel=EpsRel, limit=200)
            v2, e2 = integrate.quad(integ, qstar, qmax, epsabs=1e-10, epsrel=EpsRel, limit=200)
            vals[i], vals_err[i] = v1 + v2, e1 + e2
        else:
            vals[i], vals_err[i] = integrate.quad(integ, qmin, qmax, epsabs=1e-10, epsrel=EpsRel, limit=200)
    return vals, vals_err


def thermaldNadpT(p, m, T):
    E = np.sqrt(p**2 + m**2)
    return p/(np.exp(E/T)-1)


# pi^0 -> gamma gamma: ma = pi^0, mb = mc = gamma (massless)
ma = 0.140
T = 0.230
ini_args = (ma, T)
mb = 0.0
mc = 0.0
b = 1
pT = 10**np.linspace(-3,1, 500)
feedDown_vals = getFeedDown_anadNa(pT, ma, mb, mc, b, thermaldNadpT, ini_args)

pi_zero_thermal = thermaldNadpT(pT, ma, T)
photon_thermal = thermaldNadpT(pT, mb, T)


# Spectra comparison
plt.figure()
plt.plot(pT*1000, 2*feedDown_vals[0], color="green", label=r"$\pi^0$-feed-down")
plt.plot(pT*1000, pi_zero_thermal, color="blue", label=r"$\pi$ thermal")
plt.plot(pT*1000, photon_thermal, color="violet", label=r"$\gamma$ thermal")
plt.plot(pT*1000, photon_thermal + 2*feedDown_vals[0], label=r"$\gamma$ final")
plt.xlim(0, 2000)
plt.ylim(0.001, 1)
plt.yscale("log")
plt.legend(fontsize=13, title=r"$T = 230$ MeV", title_fontsize=13, loc="best")
plt.ylabel(r"$dN/dp_T$", fontsize=20)
plt.xlabel(r"$p_T$ [GeV]", fontsize=20)
plt.savefig("feeddown_spectra.pdf")
plt.close()

# Photon decay over pion spectra ratio
import uproot
file = uproot.open("pi0_decay_photons.root")
h_pi0 = file["h_pi0"]
h_gam = file["hDecayGam"]
values_pi0, edges =h_pi0.to_numpy()
values_gam,_ = h_gam.to_numpy()
plt.figure()
plt.stairs(values_gam/values_pi0, edges, fill=False, color='red', linewidth=2, label='Klaus Pythia')
ratio = 2*feedDown_vals[0] / pi_zero_thermal
plt.plot(pT*1000, ratio, color="green", label=r"Python formula")
plt.vlines(0.07, 0.1,50, color='gray')
print("Ratio at p_T = ", pT[0], " is ", ratio[0])
print("Ratio at p_T = ", pT[-1], " is ", ratio[-1])
plt.ylim(0.05, 50)
plt.xlim(1, 6000)
plt.xscale("log")
plt.yscale("log")
plt.legend(fontsize=13, title=r"$T = 230$ MeV", title_fontsize=13, loc="best")
plt.ylabel(r"$\gamma_{decay}/\pi^0$", fontsize=20)
plt.xlabel(r"$p_T$ [GeV]", fontsize=20)
plt.savefig("feeddown_ratio.pdf")
plt.close()
