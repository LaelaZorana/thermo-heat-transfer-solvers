"""Extended surfaces (fins): straight rectangular and pin fins.

Fin efficiency eta_f = q_fin / (h A_fin theta_b), effectiveness
eps_f = q_fin / (h A_c,base theta_b). Adiabatic tip and convective tip
formulas follow Incropera Table 3.4.

Hand check used in tests: for m L = 1 with an adiabatic tip,
eta_f = tanh(1)/1 = 0.7616. Incropera Example 3.9: aluminum pin fin,
k = 240 W/m K, D = 5 mm, L = 50 mm ... but that example is about an array,
so tests here use the closed forms directly, plus a rectangular fin case
matching Cengel Ex 3-13 style numbers (see tests).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class FinResult:
    m: float
    mL: float
    efficiency: float
    effectiveness: float
    q_fin: float
    A_fin: float
    A_base: float
    Lc: float


def _core(h, k, P, Ac, L, theta_b, tip):
    m = np.sqrt(h * P / (k * Ac))
    if tip == "adiabatic":
        Lc = L
        A_fin = P * L
        eta = np.tanh(m * L) / (m * L)
        q = np.sqrt(h * P * k * Ac) * theta_b * np.tanh(m * L)
    elif tip == "convective":
        # exact convective tip (Incropera Table 3.4 case A)
        Lc = L
        A_fin = P * L + Ac
        M = np.sqrt(h * P * k * Ac) * theta_b
        r = h / (m * k)
        q = M * (np.sinh(m * L) + r * np.cosh(m * L)) / (np.cosh(m * L) + r * np.sinh(m * L))
        eta = q / (h * A_fin * theta_b)
    elif tip == "convective_corrected":
        # corrected length approximation Lc = L + Ac/P with adiabatic formula
        Lc = L + Ac / P
        A_fin = P * Lc
        eta = np.tanh(m * Lc) / (m * Lc)
        q = eta * h * A_fin * theta_b
    else:
        raise ValueError("tip must be adiabatic, convective, or convective_corrected")
    eps = q / (h * Ac * theta_b)
    return FinResult(float(m), float(m * Lc), float(eta), float(eps), float(q),
                     float(A_fin), float(Ac), float(Lc))


def rectangular_fin(h, k, w, t, L, theta_b=1.0, tip="adiabatic") -> FinResult:
    """Straight rectangular fin of width w, thickness t, length L.
    P = 2(w + t), Ac = w t. theta_b = T_base - T_inf."""
    P = 2.0 * (w + t)
    Ac = w * t
    return _core(h, k, P, Ac, L, theta_b, tip)


def pin_fin(h, k, D, L, theta_b=1.0, tip="adiabatic") -> FinResult:
    """Cylindrical pin fin of diameter D and length L. P = pi D, Ac = pi D^2/4."""
    P = np.pi * D
    Ac = np.pi * D ** 2 / 4.0
    return _core(h, k, P, Ac, L, theta_b, tip)


def efficiency_curve(mL):
    """Adiabatic-tip fin efficiency tanh(mL)/(mL), vectorised."""
    mL = np.asarray(mL, dtype=float)
    return np.where(mL > 0, np.tanh(mL) / np.where(mL > 0, mL, 1.0), 1.0)
