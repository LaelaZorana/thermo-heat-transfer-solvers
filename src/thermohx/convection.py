"""Convection correlations with input validation and validity-range warnings.

Each function validates its inputs (Re, Ra and Pr must be positive and
real, otherwise a ValueError is raised) and issues a warnings.warn of
class RangeWarning when the inputs fall outside the range in which the
correlation was fitted. Functions accept scalar inputs; only
petukhov_friction is vectorised. Sources: Incropera chapters 7 to 9.

Hand checks used in tests:
  Dittus-Boelter, Re = 1e4, Pr = 0.7, heating: Nu = 0.023 * 1e4^0.8 * 0.7^0.4
      = 0.023 * 1584.9 * 0.8670 = 31.6
  Gnielinski, Re = 1e4, Pr = 0.7, f from Petukhov (0.790 ln Re - 1.64)^-2
      = 0.03148: Nu = (f/8)(Re-1000)Pr / (1 + 12.7 sqrt(f/8)(Pr^(2/3)-1))
      = 0.003935*9000*0.7 / (1 + 12.7*0.06273*(-0.2116)) = 24.79/0.8314 = 29.8
  Flat plate laminar, Re_L = 1e5, Pr = 0.7: Nu = 0.664 * 316.2 * 0.8879 = 186.4
  Churchill-Chu vertical plate, Ra = 1e9, Pr = 0.7: Ra^(1/6) = 31.62,
      (1 + (0.492/0.7)^(9/16))^(8/27) = 1.194, Nu = (0.825 + 10.25)^2 = 122.6
"""
from __future__ import annotations

import warnings
import numpy as np


class RangeWarning(UserWarning):
    pass


def _warn(msg):
    warnings.warn(msg, RangeWarning, stacklevel=3)


def _require_positive(**vals):
    for name, v in vals.items():
        if not v > 0:
            raise ValueError(f"{name} must be positive, got {v}")


def dittus_boelter(Re, Pr, heating=True):
    """Fully developed turbulent pipe flow, Nu = 0.023 Re^0.8 Pr^n,
    n = 0.4 heating, 0.3 cooling. Valid 0.6 <= Pr <= 160, Re >= 1e4.
    Fully developed flow, roughly L/D >= 10, is assumed and not checked."""
    _require_positive(Re=Re, Pr=Pr)
    if Re < 1e4:
        _warn(f"Dittus-Boelter: Re = {Re:.3g} < 1e4 (turbulent range only)")
    if not (0.6 <= Pr <= 160):
        _warn(f"Dittus-Boelter: Pr = {Pr:.3g} outside 0.6 to 160")
    n = 0.4 if heating else 0.3
    return 0.023 * Re ** 0.8 * Pr ** n


def petukhov_friction(Re):
    """Smooth tube friction factor, f = (0.790 ln Re - 1.64)^-2, 3000 <= Re <= 5e6.
    Vectorised; warns when any element leaves the fitted range."""
    Re = np.asarray(Re, dtype=float)
    if np.any(Re <= 0):
        raise ValueError("Re must be positive")
    if np.any(Re < 3000) or np.any(Re > 5e6):
        _warn("Petukhov friction: Re outside 3000 to 5e6")
    return (0.790 * np.log(Re) - 1.64) ** -2


def gnielinski(Re, Pr, f=None):
    """Gnielinski turbulent pipe flow, 3000 <= Re <= 5e6, 0.5 <= Pr <= 2000.
    Raises for Re <= 1000, where the formula would return a negative Nu."""
    _require_positive(Re=Re, Pr=Pr)
    if Re <= 1000:
        raise ValueError(f"Gnielinski: Re = {Re:.3g} <= 1000, correlation undefined "
                         "(the Re - 1000 factor would make Nu nonpositive)")
    if not (3000 <= Re <= 5e6):
        _warn(f"Gnielinski: Re = {Re:.3g} outside 3000 to 5e6")
    if not (0.5 <= Pr <= 2000):
        _warn(f"Gnielinski: Pr = {Pr:.3g} outside 0.5 to 2000")
    if f is None:
        f = petukhov_friction(Re)
    return (f / 8) * (Re - 1000) * Pr / (1 + 12.7 * np.sqrt(f / 8) * (Pr ** (2 / 3) - 1))


def churchill_chu_vertical_plate(Ra, Pr):
    """Natural convection on a vertical plate, all Ra (Incropera eq 9.26).
    Nu = (0.825 + 0.387 Ra^(1/6) / (1 + (0.492/Pr)^(9/16))^(8/27))^2."""
    _require_positive(Ra=Ra, Pr=Pr)
    if Ra > 1e13 or Ra < 1e-1:
        _warn(f"Churchill-Chu: Ra = {Ra:.3g} far outside the correlated range")
    return (0.825 + 0.387 * Ra ** (1 / 6) / (1 + (0.492 / Pr) ** (9 / 16)) ** (8 / 27)) ** 2


def churchill_chu_horizontal_cylinder(Ra, Pr):
    """Natural convection on a long horizontal cylinder, 1e-5 <= Ra <= 1e12."""
    _require_positive(Ra=Ra, Pr=Pr)
    if Ra > 1e12 or Ra < 1e-5:
        _warn(f"Churchill-Chu cylinder: Ra = {Ra:.3g} outside 1e-5 to 1e12")
    return (0.60 + 0.387 * Ra ** (1 / 6) / (1 + (0.559 / Pr) ** (9 / 16)) ** (8 / 27)) ** 2


def flat_plate_average(Re_L, Pr, Re_crit=5e5):
    """Average Nusselt number over an isothermal flat plate.
    Laminar (Re_L < Re_crit): 0.664 Re^0.5 Pr^(1/3), Pr >= 0.6.
    Mixed laminar plus turbulent: (0.037 Re^0.8 - A) Pr^(1/3),
    A = 0.037 Re_c^0.8 - 0.664 Re_c^0.5 (A = 871 for Re_c = 5e5),
    valid to Re_L = 1e8, 0.6 <= Pr <= 60."""
    _require_positive(Re_L=Re_L, Pr=Pr, Re_crit=Re_crit)
    if Pr < 0.6:
        _warn(f"flat plate: Pr = {Pr:.3g} < 0.6 (liquid metals not covered)")
    if Re_L < Re_crit:
        return 0.664 * Re_L ** 0.5 * Pr ** (1 / 3)
    if Re_L > 1e8 or Pr > 60:
        _warn(f"flat plate mixed: Re_L = {Re_L:.3g}, Pr = {Pr:.3g} outside range")
    A = 0.037 * Re_crit ** 0.8 - 0.664 * Re_crit ** 0.5
    return (0.037 * Re_L ** 0.8 - A) * Pr ** (1 / 3)


def flat_plate_local(Re_x, Pr, Re_crit=5e5):
    """Local Nusselt number: laminar 0.332 Re^0.5 Pr^(1/3), turbulent
    0.0296 Re^0.8 Pr^(1/3). Valid 0.6 <= Pr <= 60, Re_x <= 1e8."""
    _require_positive(Re_x=Re_x, Pr=Pr, Re_crit=Re_crit)
    if not (0.6 <= Pr <= 60):
        _warn(f"flat plate local: Pr = {Pr:.3g} outside 0.6 to 60")
    if Re_x > 1e8:
        _warn(f"flat plate local: Re_x = {Re_x:.3g} > 1e8")
    if Re_x < Re_crit:
        return 0.332 * Re_x ** 0.5 * Pr ** (1 / 3)
    return 0.0296 * Re_x ** 0.8 * Pr ** (1 / 3)
