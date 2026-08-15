"""Thermodynamic power and refrigeration cycles evaluated with CoolProp.

All properties come from CoolProp (real fluid data), so results differ
slightly from textbook table lookups (typically well under 1 percent for
efficiencies). Pressures in Pa, temperatures in K, specific quantities in
J/kg unless noted. Results are returned as dictionaries with the state
points included so callers can draw T-s diagrams.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from CoolProp.CoolProp import PropsSI


@dataclass
class State:
    """One thermodynamic state point (SI units)."""
    name: str
    p: float
    T: float
    h: float
    s: float
    x: float = -1.0   # quality, -1 if single phase

    def as_dict(self):
        return dict(name=self.name, p=self.p, T=self.T, h=self.h, s=self.s, x=self.x)


def _state(fluid, name, **kw) -> State:
    """Build a State from any two CoolProp inputs, e.g. P=..., T=... or P=..., S=..."""
    keys = list(kw.keys())
    a, b = keys[0].upper(), keys[1].upper()
    va, vb = kw[keys[0]], kw[keys[1]]
    T = PropsSI("T", a, va, b, vb, fluid)
    p = PropsSI("P", a, va, b, vb, fluid)
    h = PropsSI("H", a, va, b, vb, fluid)
    s = PropsSI("S", a, va, b, vb, fluid)
    try:
        x = PropsSI("Q", a, va, b, vb, fluid)
    except Exception:
        x = -1.0
    if not (0.0 <= x <= 1.0):
        x = -1.0
    return State(name, p, T, h, s, x)


def _expand(fluid, name, s_in, h_in, p_out, eta):
    """Isentropic expansion to p_out then apply isentropic efficiency."""
    h_s = PropsSI("H", "P", p_out, "S", s_in, fluid)
    h_out = h_in - eta * (h_in - h_s)
    return _state(fluid, name, P=p_out, H=h_out)


def _compress(fluid, name, s_in, h_in, p_out, eta):
    h_s = PropsSI("H", "P", p_out, "S", s_in, fluid)
    h_out = h_in + (h_s - h_in) / eta
    return _state(fluid, name, P=p_out, H=h_out)


# ---------------------------------------------------------------------------
# Rankine
# ---------------------------------------------------------------------------
def rankine(p_boiler, T_turbine_in, p_condenser, eta_turbine=1.0, eta_pump=1.0,
            fluid="Water") -> Dict:
    """Simple Rankine cycle: 1 sat liquid at condenser, 2 pump exit,
    3 turbine inlet, 4 turbine exit.

    Validation (Cengel, Thermodynamics, Example 10-1): 3 MPa, 350 C,
    75 kPa condenser gives eta_th = 26.0 percent. Example 10-3a: 15 MPa,
    600 C, 10 kPa gives eta_th = 43.0 percent (hand: h3 = 3583.1, h4 = 2115.3,
    w_pump = 15.1 kJ/kg, w_net = 1452.7 kJ/kg, q_in = 3376.2 kJ/kg).
    """
    s1 = _state(fluid, "1", P=p_condenser, Q=0)
    s2 = _compress(fluid, "2", s1.s, s1.h, p_boiler, eta_pump)
    s3 = _state(fluid, "3", P=p_boiler, T=T_turbine_in)
    s4 = _expand(fluid, "4", s3.s, s3.h, p_condenser, eta_turbine)
    w_pump = s2.h - s1.h
    w_turb = s3.h - s4.h
    q_in = s3.h - s2.h
    q_out = s4.h - s1.h
    w_net = w_turb - w_pump
    return dict(states=[s1, s2, s3, s4], w_turbine=w_turb, w_pump=w_pump, w_net=w_net,
                q_in=q_in, q_out=q_out, eta_th=w_net / q_in, bwr=w_pump / w_turb,
                x_turbine_exit=s4.x)


def rankine_reheat(p_boiler, T_turbine_in, p_reheat, T_reheat, p_condenser,
                   eta_turbine=1.0, eta_pump=1.0, fluid="Water") -> Dict:
    """Rankine with one reheat stage. States: 1 cond sat liq, 2 pump exit,
    3 HP turbine inlet, 4 HP exit, 5 LP inlet after reheat, 6 LP exit.

    Validation (Cengel Ex 10-4): 15 MPa, 600 C, reheat 4 MPa to 600 C,
    10 kPa: eta_th = 45.0 percent, exit quality 0.896.
    """
    s1 = _state(fluid, "1", P=p_condenser, Q=0)
    s2 = _compress(fluid, "2", s1.s, s1.h, p_boiler, eta_pump)
    s3 = _state(fluid, "3", P=p_boiler, T=T_turbine_in)
    s4 = _expand(fluid, "4", s3.s, s3.h, p_reheat, eta_turbine)
    s5 = _state(fluid, "5", P=p_reheat, T=T_reheat)
    s6 = _expand(fluid, "6", s5.s, s5.h, p_condenser, eta_turbine)
    w_pump = s2.h - s1.h
    w_turb = (s3.h - s4.h) + (s5.h - s6.h)
    q_in = (s3.h - s2.h) + (s5.h - s4.h)
    w_net = w_turb - w_pump
    return dict(states=[s1, s2, s3, s4, s5, s6], w_turbine=w_turb, w_pump=w_pump,
                w_net=w_net, q_in=q_in, q_out=s6.h - s1.h, eta_th=w_net / q_in,
                bwr=w_pump / w_turb, x_turbine_exit=s6.x)


def rankine_regenerative_open_fwh(p_boiler, T_turbine_in, p_fwh, p_condenser,
                                  eta_turbine=1.0, eta_pump=1.0, fluid="Water") -> Dict:
    """Rankine with one open feedwater heater.

    States: 1 cond sat liq, 2 pump I exit, 3 FWH exit sat liq at p_fwh,
    4 pump II exit, 5 turbine inlet, 6 bleed at p_fwh, 7 turbine exit.
    Bleed fraction y from FWH energy balance: y = (h3 - h2)/(h6 - h2).

    Validation (Cengel Ex 10-5): 15 MPa, 600 C, FWH at 1.2 MPa, 10 kPa:
    y = 0.2270, eta_th = 46.3 percent.
    """
    s1 = _state(fluid, "1", P=p_condenser, Q=0)
    s2 = _compress(fluid, "2", s1.s, s1.h, p_fwh, eta_pump)
    s3 = _state(fluid, "3", P=p_fwh, Q=0)
    s4 = _compress(fluid, "4", s3.s, s3.h, p_boiler, eta_pump)
    s5 = _state(fluid, "5", P=p_boiler, T=T_turbine_in)
    s6 = _expand(fluid, "6", s5.s, s5.h, p_fwh, eta_turbine)
    s7 = _expand(fluid, "7", s6.s, s6.h, p_condenser, eta_turbine)
    y = (s3.h - s2.h) / (s6.h - s2.h)
    q_in = s5.h - s4.h
    q_out = (1 - y) * (s7.h - s1.h)
    w_turb = (s5.h - s6.h) + (1 - y) * (s6.h - s7.h)
    w_pump = (1 - y) * (s2.h - s1.h) + (s4.h - s3.h)
    w_net = w_turb - w_pump
    return dict(states=[s1, s2, s3, s4, s5, s6, s7], y=y, w_turbine=w_turb, w_pump=w_pump,
                w_net=w_net, q_in=q_in, q_out=q_out, eta_th=w_net / q_in,
                bwr=w_pump / w_turb, x_turbine_exit=s7.x)


# ---------------------------------------------------------------------------
# Brayton
# ---------------------------------------------------------------------------
def brayton(T1, p1, pressure_ratio, T3, eta_compressor=1.0, eta_turbine=1.0,
            regenerator_effectiveness=0.0, fluid="Air") -> Dict:
    """Open air standard Brayton cycle with real air properties from CoolProp.

    States: 1 compressor inlet, 2 compressor exit, 3 turbine inlet, 4 turbine
    exit, plus 5 (regenerator exit on the cold side) when effectiveness > 0.

    Validation (Cengel Ex 9-5, variable specific heats): rp = 8, T1 = 300 K,
    T3 = 1300 K, ideal: eta_th = 42.6 percent, bwr = 0.403.
    Ex 9-6 with eta_c = 0.80, eta_t = 0.85: eta_th = 26.6 percent.
    Ex 9-7 adding a regenerator of effectiveness 0.80: eta_th = 36.9 percent.
    """
    p2 = p1 * pressure_ratio
    s1 = _state(fluid, "1", P=p1, T=T1)
    s2 = _compress(fluid, "2", s1.s, s1.h, p2, eta_compressor)
    s3 = _state(fluid, "3", P=p2, T=T3)
    s4 = _expand(fluid, "4", s3.s, s3.h, p1, eta_turbine)
    w_c = s2.h - s1.h
    w_t = s3.h - s4.h
    states = [s1, s2, s3, s4]
    h_in_combustor = s2.h
    if regenerator_effectiveness > 0:
        h5 = s2.h + regenerator_effectiveness * (s4.h - s2.h)
        s5 = _state(fluid, "5", P=p2, H=h5)
        states.append(s5)
        h_in_combustor = h5
    q_in = s3.h - h_in_combustor
    w_net = w_t - w_c
    return dict(states=states, w_turbine=w_t, w_compressor=w_c, w_net=w_net, q_in=q_in,
                eta_th=w_net / q_in, bwr=w_c / w_t)


# ---------------------------------------------------------------------------
# Vapor compression refrigeration
# ---------------------------------------------------------------------------
def vapor_compression(p_evap, p_cond, eta_compressor=1.0, superheat_K=0.0,
                      subcool_K=0.0, fluid="R134a") -> Dict:
    """Vapor compression cycle: 1 compressor inlet (sat vapor plus optional
    superheat), 2 compressor exit, 3 condenser exit (sat liquid plus optional
    subcooling), 4 after throttling (h4 = h3).

    Validation (Cengel Ex 11-1): R134a, 0.14 MPa to 0.8 MPa, ideal:
    q_L = 143.7 kJ/kg, w_in = 36.2 kJ/kg, COP_R = 3.97 (table values,
    CoolProp gives about 3.97 as well).
    """
    if superheat_K > 0:
        Tsat = PropsSI("T", "P", p_evap, "Q", 1, fluid)
        s1 = _state(fluid, "1", P=p_evap, T=Tsat + superheat_K)
    else:
        s1 = _state(fluid, "1", P=p_evap, Q=1)
    s2 = _compress(fluid, "2", s1.s, s1.h, p_cond, eta_compressor)
    if subcool_K > 0:
        Tsat = PropsSI("T", "P", p_cond, "Q", 0, fluid)
        s3 = _state(fluid, "3", P=p_cond, T=Tsat - subcool_K)
    else:
        s3 = _state(fluid, "3", P=p_cond, Q=0)
    s4 = _state(fluid, "4", P=p_evap, H=s3.h)
    q_L = s1.h - s4.h
    q_H = s2.h - s3.h
    w_in = s2.h - s1.h
    return dict(states=[s1, s2, s3, s4], q_L=q_L, q_H=q_H, w_in=w_in,
                COP_R=q_L / w_in, COP_HP=q_H / w_in)


def saturation_dome(fluid="Water", n=200):
    """Return (s_liq, T_liq, s_vap, T_vap) arrays for plotting a T-s dome."""
    import numpy as np
    Tt = PropsSI("Ttriple", fluid) + 0.5
    Tc = PropsSI("Tcrit", fluid) - 0.05
    T = np.linspace(Tt, Tc, n)
    sl = np.array([PropsSI("S", "T", t, "Q", 0, fluid) for t in T])
    sv = np.array([PropsSI("S", "T", t, "Q", 1, fluid) for t in T])
    return sl, T, sv, T
