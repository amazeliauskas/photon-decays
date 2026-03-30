#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy.integrate as integrate
from scipy.special import ellipk
from scipy.special import ellipkinc
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
            if ((am + ap)/(ap*am + am + ap + 1)) < 0:
                if Debug:
                    print("Warning! Negative sqrt found in argument of arcsin. Returning NaN.")
                raise ValueError()
            argument = np.sqrt(2)*np.sqrt((am + ap)/(ap*am + am + ap + 1))
            if (argument < -1) or (argument < 1):
                if Debug:
                    print("Warning: Argument outside of valid range of arcsin. Returning NaN.")
                raise ValueError()
            phi = np.arcsin(argument)
            m = ((am+1)*(ap+1)/(2*(am + ap)))**2
            denom = -1*(ap - 1)*(am + ap)/(am+1)**2
            if ((-am-1) < 0) or ((ap - 1) < 0) or (denom < 0):
                if Debug:
                    print("Warning! Negative sqrt found in phiIntegral. Returning NaN.")
                raise ValueError()
            phiIntegral = np.sqrt(2)*np.sqrt(-am - 1)*(-1*(am + 1))**(-3/2)*np.sqrt(ap-1)*ellipkinc(phi, m)/np.sqrt(denom)
        else:
            argument = (1 - am)*(ap-1)/(2*(am+ap))
            elliptick = ellipk(argument)
            if (((am + 1)/(am + ap)) < 0) or ((am+1) < 0):
                if Debug:
                    print("Warning! Negative sqrt found in phiIntegral. Returning NaN.")
                raise ValueError()
            phiIntegral = np.sqrt(2)*np.sqrt((am + 1)/(am + ap))/np.sqrt(am+1)*elliptick
        return norm/pT/qT*phiIntegral
    except ValueError:
        return np.nan
    except ZeroDivisionError:
        if Debug:
            print("Warning! Division by zero, returning NaN")
        return np.nan

def integrand_factor_massless(qT, b, ma, mb, mc, pT, Debug=False):
    try:
        norm = 2*b/np.pi
        am = getAminus(qT, ma, mb, mc, pT)
        ap = getAplus(qT, ma, mb, mc, pT)
        argument = (1 - am)*(ap-1)/(2*(am+ap))
        elliptick = ellipk(argument)
        if ((am+1)/(am + ap) < 0) or ((am+1) < 0):
            if Debug:
                print("Warning! Negative sqrt found in phiIntegral. Returning NaN.")
            raise ValueError()
        phiIntegral = np.sqrt(2)*np.sqrt((am+1)/(am + ap))/np.sqrt(am+1)*elliptick
        return norm*phiIntegral/qT
    except ValueError:
        return np.nan
    except ZeroDivisionError:
        if Debug:
            print("Warning! Division by zero, returning NaN")
        return np.nan


def getFeedDown_anadNa(pT, ma, mb, mc, b, dNa_dpT, dNa_dpT_args, EpsRel=1e-4, Debug=False):
    vals = np.zeros_like(pT)
    vals_err = np.zeros_like(pT)
    for i, p in enumerate(pT):
        qmin = max(get_qTminus(ma, mb, mc, p), 0)
        qmax = get_qTplus(ma, mb, mc, p)
        args = (b, ma, mb, mc, p, Debug)
        if (mb == 0) and (mc == 0):
            integ = lambda qT: integrand_factor_massless(qT, *args)*dNa_dpT(qT, *dNa_dpT_args)
            vals[i], vals_err[i] = integrate.quad(integ, qmin, qmax, epsabs=1e-10, epsrel=EpsRel, limit=200)
        else:
            integ = lambda qT: integrand_factor_massive(qT, *args)*dNa_dpT(qT, *dNa_dpT_args)
            vals[i], vals_err[i] = integrate.quad(integ, qmin, qmax, epsabs=1e-10, epsrel=EpsRel, limit=200)
    return vals, vals_err


def thermaldNadpT(p, m, T):
    E = np.sqrt(p**2 + m**2)
    return p/(np.exp(E/T)-1)


# pi^0 -> gamma gamma: ma = pi^0, mb = mc = gamma (massless)
ma = 0.14
T = 0.230
ini_args = (ma, T)
mb = 0.0
mc = 0.0
b = 1
pT = np.linspace(0.1, 2, 500)
feedDown_vals = getFeedDown_anadNa(pT, ma, mb, mc, b, thermaldNadpT, ini_args)

pi_zero_thermal = thermaldNadpT(pT, ma, T)
photon_thermal = thermaldNadpT(pT, mb, T)

# Spectra comparison
plt.figure()
plt.plot(pT, 2*feedDown_vals[0], color="green", label=r"$\pi^0$-feed-down")
plt.plot(pT, pi_zero_thermal, color="blue", label=r"$\pi$ thermal")
plt.plot(pT, photon_thermal, color="violet", label=r"$\gamma$ thermal")
plt.plot(pT, photon_thermal + 2*feedDown_vals[0], label=r"$\gamma$ final")
plt.xlim(0.09, 2)
plt.yscale("log")
plt.legend(fontsize=13, title=r"$T = 230$ MeV", title_fontsize=13, loc="best")
plt.ylabel(r"$dN/dp_T$", fontsize=20)
plt.xlabel(r"$p_T$ [GeV]", fontsize=20)
plt.savefig("feeddown_spectra.pdf")
plt.close()

# Photon decay over pion spectra ratio
ratio = 2*feedDown_vals[0] / pi_zero_thermal
plt.figure()
plt.plot(pT, ratio, color="green", label=r"$\pi^0$-Feed-down")
print("Ratio at p_T = ", pT[0], " is ", ratio[0])
print("Ratio at p_T = ", pT[-1], " is ", ratio[-1])
plt.ylim(0.1, 50)
plt.xlim(0, 2)
plt.yscale("log")
plt.legend(fontsize=13, title=r"$T = 230$ MeV", title_fontsize=13, loc="best")
plt.ylabel(r"$\gamma_{decay}/\pi^0$", fontsize=20)
plt.xlabel(r"$p_T$ [GeV]", fontsize=20)
plt.savefig("feeddown_ratio.pdf")
plt.close()
