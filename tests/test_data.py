"""Data path tests: property tables against CoolProp, and the heat
exchanger and cycle spec loaders for internal consistency."""
import numpy as np
import pytest
from CoolProp.CoolProp import PropsSI

from thermohx import hx, properties, specs

# Per property agreement between the tables and CoolProp, as fractions.
TOL = {"rho_kg_m3": 0.01, "cp_J_kgK": 0.01, "mu_Pa_s": 0.025,
       "k_W_mK": 0.025, "Pr": 0.035}


def coolprop_args(fluid, T):
    if fluid == "air":
        return ("T", T, "P", 1e5, "Air")
    return ("T", T, "Q", 0, "Water")


@pytest.mark.parametrize("fluid", ["air", "water"])
def test_table_matches_coolprop(fluid):
    table = properties.load_table(fluid)
    for i, T in enumerate(table["T_K"]):
        args = coolprop_args(fluid, T)
        ref = {"rho_kg_m3": PropsSI("D", *args), "cp_J_kgK": PropsSI("C", *args),
               "mu_Pa_s": PropsSI("V", *args), "k_W_mK": PropsSI("L", *args)}
        ref["Pr"] = ref["mu_Pa_s"] * ref["cp_J_kgK"] / ref["k_W_mK"]
        for name, tol in TOL.items():
            err = abs(table[name][i] - ref[name]) / ref[name]
            assert err < tol, (fluid, T, name, err)


@pytest.mark.parametrize("fluid", ["air", "water"])
def test_interpolation_hits_nodes_and_between(fluid):
    table = properties.load_table(fluid)
    T0, T1 = table["T_K"][0], table["T_K"][1]
    assert properties.prop(fluid, T0, "cp_J_kgK") == table["cp_J_kgK"][0]
    mid = properties.prop(fluid, 0.5 * (T0 + T1), "cp_J_kgK")
    lo, hi = sorted([table["cp_J_kgK"][0], table["cp_J_kgK"][1]])
    assert lo <= mid <= hi
    with pytest.raises(ValueError):
        properties.prop(fluid, T0 - 50, "cp_J_kgK")
    with pytest.raises(ValueError):
        properties.prop(fluid, 5000, "cp_J_kgK")


def test_interpolated_prop_close_to_coolprop_between_nodes():
    # The fallback path should track CoolProp between nodes too.
    for fluid, T in [("air", 325.0), ("air", 725.0), ("water", 340.0)]:
        args = coolprop_args(fluid, T)
        assert properties.prop(fluid, T, "cp_J_kgK") == pytest.approx(
            PropsSI("C", *args), rel=0.01)
        assert properties.prop(fluid, T, "k_W_mK") == pytest.approx(
            PropsSI("L", *args), rel=0.03)


def test_hx_specs_rate_consistently():
    spec_dir = specs.DATA_DIR / "heat_exchangers"
    paths = sorted(spec_dir.glob("*.yaml"))
    assert len(paths) >= 3
    for path in paths:
        spec = specs.load_spec(path)
        r, info = specs.rate_hx_spec(spec)
        assert r.q > 0, path.name
        assert 0 < r.effectiveness < 1
        # energy balance: both streams carry the same duty
        assert info["Ch"] * (float(spec["hot"]["T_in_K"]) - r.Th_out) == \
            pytest.approx(r.q, rel=1e-9)
        assert info["Cc"] * (r.Tc_out - float(spec["cold"]["T_in_K"])) == \
            pytest.approx(r.q, rel=1e-9)
        assert r.Th_out > r.Tc_out or spec["arrangement"] == "counter"


def test_radiator_duty_plausible():
    spec = specs.load_spec(specs.DATA_DIR / "heat_exchangers" /
                           "car_radiator_crossflow.yaml")
    r, _ = specs.rate_hx_spec(spec)
    assert 30e3 < r.q < 120e3   # tens of kW for a passenger car radiator


def test_steam_plant_spec():
    spec = specs.load_spec(specs.DATA_DIR / "cycles" / "steam_plant_500mw.yaml")
    r = specs.run_cycle_spec(spec)
    # first law closure and a plausible utility plant efficiency
    assert r["w_net"] == pytest.approx(r["q_in"] - r["q_out"], rel=1e-9)
    assert 0.36 < r["eta_th"] < 0.48
    assert 0 < r["y"] < 0.35
    assert 0.80 < r["x_turbine_exit"] <= 1.0 or r["x_turbine_exit"] == -1.0
    assert 250 < r["m_dot_kg_s"] < 600
    assert r["heat_input_W"] == pytest.approx(500e6 / r["eta_th"], rel=1e-9)


def test_gas_turbine_spec():
    spec = specs.load_spec(specs.DATA_DIR / "cycles" /
                           "gas_turbine_5mw_recuperated.yaml")
    r = specs.run_cycle_spec(spec)
    assert 0.30 < r["eta_th"] < 0.50
    st = {s.name: s for s in r["states"]}
    assert st["4"].T > st["2"].T        # exhaust hotter than compressor exit
    assert st["2"].T < st["5"].T < st["4"].T
    assert 5 < r["m_dot_kg_s"] < 40
    # unrecuperated version of the same machine must be less efficient
    plain = dict(spec, inputs=dict(spec["inputs"], regenerator_effectiveness=0))
    r0 = specs.run_cycle_spec(plain)
    assert r0["eta_th"] < r["eta_th"]
