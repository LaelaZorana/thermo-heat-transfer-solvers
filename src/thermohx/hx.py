"""Heat exchanger analysis: LMTD method and effectiveness-NTU method.

Supported flow arrangements: "parallel", "counter", "shell_tube_1_2"
(one shell pass, 2, 4, ... tube passes), "cross_unmixed" (both fluids
unmixed). Effectiveness relations follow Incropera Table 11.3 and their
inverses Table 11.4 (crossflow unmixed is inverted numerically).

Two modes are provided:
  size(): given inlet temperatures, capacity rates, and the required duty
          (or one outlet temperature) plus U, return the area via the
          eps-NTU route; LMTD and F are computed afterward for reporting.
  rate(): given U A, inlet temperatures and capacity rates, return outlet
          temperatures and duty (eps-NTU route).

Validation (Incropera Example 11.1): counterflow oil cooler, oil 0.1 kg/s
cp 2131 from 100 C to 60 C, water 0.2 kg/s cp 4178 in at 30 C, U = 38.1
W/m2 K. Hand: q = 8524 W, Tc,o = 40.2 C, LMTD = 43.2 C, A = 5.18 m2,
which for a 25 mm tube gives L = 65.9 m.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq

ARRANGEMENTS = ("parallel", "counter", "shell_tube_1_2", "cross_unmixed")


def lmtd(Th_in, Th_out, Tc_in, Tc_out, arrangement="counter"):
    """Log mean temperature difference. For counterflow style endpoints
    dT1 = Th_in - Tc_out, dT2 = Th_out - Tc_in; parallel uses
    dT1 = Th_in - Tc_in, dT2 = Th_out - Tc_out. Other arrangements return
    the counterflow LMTD (apply F separately)."""
    if arrangement == "parallel":
        d1, d2 = Th_in - Tc_in, Th_out - Tc_out
    else:
        d1, d2 = Th_in - Tc_out, Th_out - Tc_in
    if d1 <= 0 or d2 <= 0:
        raise ValueError("temperature cross: LMTD undefined for these endpoints")
    if abs(d1 - d2) < 1e-9 * max(d1, d2):
        return d1
    return (d1 - d2) / np.log(d1 / d2)


def correction_factor_F(Th_in, Th_out, Tc_in, Tc_out, arrangement):
    """LMTD correction factor F. Shell-and-tube 1-2: closed form
    (Incropera Fig 11.10 equation). Crossflow unmixed: computed from the
    eps-NTU relation so that F = NTU_counter / NTU_actual at equal eps."""
    if arrangement in ("parallel", "counter"):
        return 1.0
    # Since Ch dTh = Cc dTc = q, the fluid with the larger temperature change
    # is Cmin, Cr = dT_small/dT_large, and eps = dT_large/(Th_in - Tc_in).
    dTh, dTc = Th_in - Th_out, Tc_out - Tc_in
    big, small = max(dTh, dTc), min(dTh, dTc)
    if small <= 0:
        # One stream is isothermal (Cr = 0): every arrangement collapses to
        # the same eps-NTU relation, so F = 1 exactly.
        return 1.0
    eps = big / (Th_in - Tc_in)
    Cr = small / big
    ntu_actual = ntu_from_effectiveness(eps, Cr, arrangement)
    ntu_cf = ntu_from_effectiveness(eps, Cr, "counter")
    return ntu_cf / ntu_actual


def effectiveness(NTU, Cr, arrangement="counter"):
    """Effectiveness as a function of NTU and Cr = Cmin/Cmax."""
    NTU = float(NTU)
    Cr = float(Cr)
    if NTU < 0:
        raise ValueError(f"NTU must be nonnegative, got {NTU}")
    if not 0 <= Cr <= 1:
        raise ValueError(f"Cr must lie in [0, 1], got {Cr}")
    if NTU == 0:
        return 0.0
    if Cr < 1e-12:
        return 1.0 - np.exp(-NTU)
    if arrangement == "parallel":
        return (1 - np.exp(-NTU * (1 + Cr))) / (1 + Cr)
    if arrangement == "counter":
        if abs(Cr - 1) < 1e-9:
            return NTU / (1 + NTU)
        e = np.exp(-NTU * (1 - Cr))
        return (1 - e) / (1 - Cr * e)
    if arrangement == "shell_tube_1_2":
        r = np.sqrt(1 + Cr ** 2)
        e = np.exp(-NTU * r)
        return 2.0 / (1 + Cr + r * (1 + e) / (1 - e))
    if arrangement == "cross_unmixed":
        return 1 - np.exp((1 / Cr) * NTU ** 0.22 * (np.exp(-Cr * NTU ** 0.78) - 1))
    raise ValueError(f"unknown arrangement {arrangement}")


def ntu_from_effectiveness(eps, Cr, arrangement="counter"):
    """Inverse relations (Incropera Table 11.4); crossflow solved numerically.
    Raises ValueError when eps is negative or exceeds the arrangement's
    limiting effectiveness at the given Cr."""
    eps = float(eps)
    Cr = float(Cr)
    if not 0 <= Cr <= 1:
        raise ValueError(f"Cr must lie in [0, 1], got {Cr}")
    if eps < 0:
        raise ValueError(f"effectiveness must be nonnegative, got {eps}")
    if eps == 0:
        return 0.0
    limit = max_effectiveness(Cr, arrangement)
    if eps >= limit:
        raise ValueError(f"effectiveness {eps:.4g} is not attainable for "
                         f"{arrangement} at Cr = {Cr:.4g} (limit {limit:.4g})")
    if Cr < 1e-12:
        return -np.log(1 - eps)
    if arrangement == "parallel":
        return -np.log(1 - eps * (1 + Cr)) / (1 + Cr)
    if arrangement == "counter":
        if abs(Cr - 1) < 1e-9:
            return eps / (1 - eps)
        return np.log((eps - 1) / (eps * Cr - 1)) / (Cr - 1)
    if arrangement == "shell_tube_1_2":
        r = np.sqrt(1 + Cr ** 2)
        E = (2 / eps - (1 + Cr)) / r
        return -np.log((E - 1) / (E + 1)) / r
    if arrangement == "cross_unmixed":
        f = lambda n: effectiveness(n, Cr, arrangement) - eps
        if f(200.0) < 0:
            raise ValueError("requested effectiveness not attainable for crossflow unmixed")
        return brentq(f, 1e-9, 200.0)
    raise ValueError(f"unknown arrangement {arrangement}")


def max_effectiveness(Cr, arrangement="counter"):
    """Limiting effectiveness as NTU grows without bound, per arrangement."""
    Cr = float(Cr)
    if not 0 <= Cr <= 1:
        raise ValueError(f"Cr must lie in [0, 1], got {Cr}")
    if Cr < 1e-12:
        return 1.0
    if arrangement == "parallel":
        return 1.0 / (1 + Cr)
    if arrangement == "counter":
        return 1.0
    if arrangement == "shell_tube_1_2":
        return 2.0 / (1 + Cr + np.sqrt(1 + Cr ** 2))
    if arrangement == "cross_unmixed":
        return min(1.0, effectiveness(200.0, Cr, arrangement))
    raise ValueError(f"unknown arrangement {arrangement}")


@dataclass
class HXResult:
    q: float
    Th_out: float
    Tc_out: float
    UA: float
    area: float
    NTU: float
    effectiveness: float
    Cr: float
    lmtd: float
    F: float


def size(Ch, Cc, Th_in, Tc_in, U, q=None, Th_out=None, Tc_out=None,
         arrangement="counter") -> HXResult:
    """Sizing: return the area for a given duty. Supply exactly one of q,
    Th_out, Tc_out. Ch, Cc are capacity rates (m_dot cp) in W/K."""
    if Ch <= 0 or Cc <= 0 or U <= 0:
        raise ValueError("Ch, Cc and U must all be positive")
    if Th_in <= Tc_in:
        raise ValueError("Th_in must exceed Tc_in")
    given = [v is not None for v in (q, Th_out, Tc_out)]
    if sum(given) != 1:
        raise ValueError("supply exactly one of q, Th_out, Tc_out")
    if q is None:
        q = Ch * (Th_in - Th_out) if Th_out is not None else Cc * (Tc_out - Tc_in)
    if q <= 0:
        raise ValueError(f"duty must be positive, got q = {q}")
    Th_out = Th_in - q / Ch
    Tc_out = Tc_in + q / Cc
    Cmin, Cmax = min(Ch, Cc), max(Ch, Cc)
    Cr = Cmin / Cmax
    eps = q / (Cmin * (Th_in - Tc_in))
    NTU = ntu_from_effectiveness(eps, Cr, arrangement)
    UA = NTU * Cmin
    dTlm = lmtd(Th_in, Th_out, Tc_in, Tc_out,
                "parallel" if arrangement == "parallel" else "counter")
    F = correction_factor_F(Th_in, Th_out, Tc_in, Tc_out, arrangement)
    return HXResult(q, Th_out, Tc_out, UA, UA / U, NTU, eps, Cr, dTlm, F)


def rate(Ch, Cc, Th_in, Tc_in, UA, arrangement="counter") -> HXResult:
    """Rating: given UA, return outlet temperatures and duty (eps-NTU).
    The area field of the result is nan because rating knows only the UA
    product, not U and A separately."""
    if Ch <= 0 or Cc <= 0 or UA <= 0:
        raise ValueError("Ch, Cc and UA must all be positive")
    if Th_in <= Tc_in:
        raise ValueError("Th_in must exceed Tc_in")
    Cmin, Cmax = min(Ch, Cc), max(Ch, Cc)
    Cr = Cmin / Cmax
    NTU = UA / Cmin
    eps = effectiveness(NTU, Cr, arrangement)
    q = eps * Cmin * (Th_in - Tc_in)
    Th_out = Th_in - q / Ch
    Tc_out = Tc_in + q / Cc
    dTlm = lmtd(Th_in, Th_out, Tc_in, Tc_out,
                "parallel" if arrangement == "parallel" else "counter")
    # F straight from the NTU already in hand; the inverse eps-NTU call can
    # fail near the limiting effectiveness, and a rating must not fail on a
    # diagnostic, so fall back to nan.
    if arrangement in ("parallel", "counter") or Cr < 1e-12:
        F = 1.0
    else:
        try:
            F = ntu_from_effectiveness(eps, Cr, "counter") / NTU
        except ValueError:
            F = float("nan")
    return HXResult(q, Th_out, Tc_out, UA, float("nan"), NTU, eps, Cr, dTlm, F)
