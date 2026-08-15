"""Loaders for the YAML equipment specs in data/heat_exchangers/ and
data/cycles/, plus the routines that rate or run them.

A heat exchanger spec gives the arrangement, U and area, and for each
stream the fluid, mass flow and inlet temperature. Specific heats come
from the tables in data/fluids/ evaluated at the mean of inlet and outlet
(one fixed point iteration), or from a cp_J_kgK override in the spec for
fluids not tabulated, such as oil. Rating goes through thermohx.hx.rate.

A cycle spec names one of the cycle functions in thermohx.cycles, carries
its keyword arguments, and optionally a net electrical or shaft power from
which the mass flow follows as net_power_W / w_net.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from . import cycles, hx, properties

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_spec(path):
    """Load one YAML spec file as a dict."""
    with open(path) as f:
        spec = yaml.safe_load(f)
    if not isinstance(spec, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    spec["_path"] = str(path)
    return spec


def _stream_cp(stream, T_mean):
    if "cp_J_kgK" in stream:
        return float(stream["cp_J_kgK"])
    return properties.prop(stream["fluid"], T_mean, "cp_J_kgK")


def rate_hx_spec(spec):
    """Rate one heat exchanger spec. Returns (HXResult, info dict with the
    capacity rates and the cp values actually used)."""
    hot, cold = spec["hot"], spec["cold"]
    Th_in, Tc_in = float(hot["T_in_K"]), float(cold["T_in_K"])
    UA = float(spec["U_W_m2K"]) * float(spec["area_m2"])
    # First pass with cp at the inlets, then one refinement at the means.
    cph, cpc = _stream_cp(hot, Th_in), _stream_cp(cold, Tc_in)
    for _ in range(2):
        Ch = float(hot["m_dot_kg_s"]) * cph
        Cc = float(cold["m_dot_kg_s"]) * cpc
        r = hx.rate(Ch, Cc, Th_in, Tc_in, UA, spec["arrangement"])
        cph = _stream_cp(hot, 0.5 * (Th_in + r.Th_out))
        cpc = _stream_cp(cold, 0.5 * (Tc_in + r.Tc_out))
    info = dict(Ch=Ch, Cc=Cc, cp_hot=cph, cp_cold=cpc, UA=UA)
    return r, info


def run_cycle_spec(spec):
    """Run one cycle spec. Returns the cycle result dict with m_dot_kg_s,
    net_power_W and heat_input_W added when the spec gives a net power."""
    fn = getattr(cycles, spec["cycle"], None)
    if fn is None or spec["cycle"].startswith("_"):
        raise ValueError(f"unknown cycle function {spec['cycle']!r}")
    # YAML 1.1 reads exponent forms like 16.5e6 as strings, so cast here.
    inputs = {k: float(v) for k, v in spec.get("inputs", {}).items()}
    result = fn(**inputs)
    if "net_power_W" in spec:
        P = float(spec["net_power_W"])
        m_dot = P / result["w_net"]
        result["m_dot_kg_s"] = m_dot
        result["net_power_W"] = P
        result["heat_input_W"] = m_dot * result["q_in"]
    return result
