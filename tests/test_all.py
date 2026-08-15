"""Validation tests against textbook cases (Cengel Thermodynamics 8e,
Cengel Heat Transfer, Incropera 7e). Textbook numbers were recomputed by
hand from the tables and are cited in the module docstrings; CoolProp
real-fluid data is expected to land within about 1 percent of table values."""
import warnings
import numpy as np
import pytest

from thermohx import cycles, fins, hx, transient, convection

MPa = 1e6
kPa = 1e3


def rel(a, b):
    return abs(a - b) / abs(b)


# ------------------------------------------------------------------ cycles
def test_rankine_cengel_10_1():
    r = cycles.rankine(3 * MPa, 350 + 273.15, 75 * kPa)
    assert rel(r["eta_th"], 0.260) < 0.01          # Cengel: 26.0 percent
    assert 0 < r["bwr"] < 0.01
    assert r["x_turbine_exit"] == pytest.approx(0.886, abs=0.005)


def test_rankine_cengel_10_3():
    r = cycles.rankine(15 * MPa, 600 + 273.15, 10 * kPa)
    assert rel(r["eta_th"], 0.430) < 0.01          # Cengel: 43.0 percent
    assert rel(r["w_net"], 1452.7e3) < 0.005       # hand from tables


def test_rankine_real_turbine_pump():
    r = cycles.rankine(15 * MPa, 600 + 273.15, 10 * kPa, eta_turbine=0.87, eta_pump=0.85)
    ideal = cycles.rankine(15 * MPa, 600 + 273.15, 10 * kPa)
    assert r["eta_th"] < ideal["eta_th"]
    assert r["w_pump"] > ideal["w_pump"]
    assert r["states"][3].s > ideal["states"][3].s   # entropy generation


def test_rankine_reheat_cengel_10_4():
    r = cycles.rankine_reheat(15 * MPa, 873.15, 4 * MPa, 873.15, 10 * kPa)
    assert rel(r["eta_th"], 0.450) < 0.01          # Cengel: 45.0 percent
    assert r["x_turbine_exit"] == pytest.approx(0.896, abs=0.005)


def test_rankine_open_fwh_cengel_10_5():
    r = cycles.rankine_regenerative_open_fwh(15 * MPa, 873.15, 1.2 * MPa, 10 * kPa)
    assert r["y"] == pytest.approx(0.2270, abs=0.002)   # Cengel: y = 0.2270
    assert rel(r["eta_th"], 0.463) < 0.01               # Cengel: 46.3 percent


def test_brayton_cengel_9_5_6_7():
    b = cycles.brayton(300, 100 * kPa, 8, 1300)
    assert rel(b["eta_th"], 0.426) < 0.01          # Cengel: 42.6 percent
    assert rel(b["bwr"], 0.403) < 0.01
    b2 = cycles.brayton(300, 100 * kPa, 8, 1300, 0.80, 0.85)
    assert rel(b2["eta_th"], 0.266) < 0.01         # Cengel: 26.6 percent
    b3 = cycles.brayton(300, 100 * kPa, 8, 1300, 0.80, 0.85, regenerator_effectiveness=0.80)
    assert rel(b3["eta_th"], 0.369) < 0.01         # Cengel: 36.9 percent


def test_vapor_compression_cengel_11_1():
    v = cycles.vapor_compression(0.14 * MPa, 0.8 * MPa)
    assert rel(v["COP_R"], 3.97) < 0.01
    assert rel(v["q_L"], 143.7e3) < 0.01
    assert rel(v["w_in"], 36.2e3) < 0.01
    assert v["COP_HP"] == pytest.approx(v["COP_R"] + 1, rel=1e-9)


# ------------------------------------------------------------------- fins
def test_fin_efficiency_closed_form():
    # choose geometry so that mL = 1: eta = tanh(1) = 0.7616
    h, k, D = 100.0, 200.0, 0.01
    m = np.sqrt(4 * h / (k * D))            # pin fin: hP/(kAc) = 4h/(kD)
    L = 1.0 / m
    r = fins.pin_fin(h, k, D, L, tip="adiabatic")
    assert r.efficiency == pytest.approx(np.tanh(1.0), rel=1e-9)
    assert r.mL == pytest.approx(1.0)


def test_rectangular_fin_hand_case():
    # Aluminum fin k = 180, w = 1 m, t = 3 mm, L = 30 mm, h = 25, theta_b = 60 K.
    # m = sqrt(hP/kAc) = sqrt(25*2*(1.003)/(180*0.003)) = 9.636 1/m, mL = 0.2891,
    # eta (adiabatic) = tanh(0.2891)/0.2891 = 0.9729,
    # q = sqrt(hPkAc) theta_b tanh(mL) = sqrt(25*2.006*180*0.003)*60*0.2813
    #   = 5.203*60*0.2813 = 87.8 W per m of width.
    r = fins.rectangular_fin(25, 180, 1.0, 0.003, 0.03, theta_b=60, tip="adiabatic")
    assert r.efficiency == pytest.approx(0.9729, abs=1e-3)
    assert r.q_fin == pytest.approx(87.8, rel=0.005)
    # convective tip must give slightly more heat than adiabatic, and the
    # corrected-length shortcut should agree with the exact convective tip
    rc = fins.rectangular_fin(25, 180, 1.0, 0.003, 0.03, theta_b=60, tip="convective")
    rcc = fins.rectangular_fin(25, 180, 1.0, 0.003, 0.03, theta_b=60, tip="convective_corrected")
    assert rc.q_fin > r.q_fin
    assert rc.q_fin == pytest.approx(rcc.q_fin, rel=0.005)


def test_fin_effectiveness_criterion():
    # Effectiveness of a long fin = sqrt(kP/(hAc)); a fin only pays when > 2
    r = fins.pin_fin(10.0, 200.0, 0.005, 0.5, tip="adiabatic")
    long_fin_eps = np.sqrt(200 * np.pi * 0.005 / (10 * np.pi * 0.005 ** 2 / 4))
    assert r.effectiveness == pytest.approx(long_fin_eps, rel=5e-3)   # tanh(3.16) = 0.996
    assert r.effectiveness > 2


# --------------------------------------------------------------------- hx
def test_hx_incropera_11_1_counterflow_size():
    Ch, Cc = 0.1 * 2131, 0.2 * 4178
    s = hx.size(Ch, Cc, 100, 30, U=38.1, Th_out=60, arrangement="counter")
    assert s.q == pytest.approx(8524, rel=1e-6)
    assert s.Tc_out == pytest.approx(40.2, abs=0.05)
    assert s.lmtd == pytest.approx(43.2, abs=0.05)
    assert s.area == pytest.approx(5.18, abs=0.02)
    assert s.area / (np.pi * 0.025) == pytest.approx(65.9, abs=0.2)
    assert s.F == pytest.approx(1.0)


def test_hx_ntu_and_lmtd_agree():
    # LMTD area = q/(U F LMTD) must equal NTU Cmin/U for every arrangement
    Ch, Cc = 2000.0, 3000.0
    for arr in hx.ARRANGEMENTS:
        s = hx.size(Ch, Cc, 150, 20, U=100, q=100e3, arrangement=arr)
        A_lmtd = s.q / (100 * s.F * s.lmtd)
        assert A_lmtd == pytest.approx(s.area, rel=1e-6), arr


def test_hx_rate_roundtrip():
    Ch, Cc = 1500.0, 2500.0
    for arr in hx.ARRANGEMENTS:
        s = hx.size(Ch, Cc, 120, 25, U=250, Tc_out=55, arrangement=arr)
        r = hx.rate(Ch, Cc, 120, 25, UA=s.UA, arrangement=arr)
        assert r.Tc_out == pytest.approx(55, abs=1e-6), arr
        assert r.q == pytest.approx(s.q, rel=1e-8), arr


def test_hx_effectiveness_limits():
    assert hx.effectiveness(2.0, 1.0, "counter") == pytest.approx(2 / 3)
    assert hx.effectiveness(2.0, 0.0, "parallel") == pytest.approx(1 - np.exp(-2))
    # counterflow beats parallel beats nothing else at the same NTU, Cr
    e = {a: hx.effectiveness(3.0, 0.8, a) for a in hx.ARRANGEMENTS}
    assert e["counter"] > e["cross_unmixed"] > e["shell_tube_1_2"] > e["parallel"]
    # inverse relations round trip
    for a in hx.ARRANGEMENTS:
        assert hx.ntu_from_effectiveness(e[a], 0.8, a) == pytest.approx(3.0, rel=1e-6)


def test_hx_shell_tube_F_hand_value():
    # Th 100 to 60, Tc 30 to 40.2: P = 0.1457, R = 3.92. Bowman closed form
    # F = S ln((1-P)/(1-PR)) / ((R-1) ln((2-P(R+1-S))/(2-P(R+1+S)))), S = sqrt(R^2+1)
    # evaluates by hand to 0.9615.
    F = hx.correction_factor_F(100, 60, 30, 40.2, "shell_tube_1_2")
    assert F == pytest.approx(0.9615, abs=5e-4)


# --------------------------------------------------------------- transient
def test_lumped_cengel_4_1():
    D = 1e-3
    V, A = np.pi * D ** 3 / 6, np.pi * D ** 2
    r = transient.lumped_capacitance(25, 200, 210, 8500, 320, 35, V, A, t=[0, 10])
    assert r.Bi == pytest.approx(0.001, rel=1e-6)
    assert r.valid
    assert r.tau == pytest.approx(2.159, abs=0.002)
    t99 = transient.lumped_time_to_reach(25 + 0.99 * 175, 25, 200, 210, 8500, 320, V, A)
    assert t99 == pytest.approx(9.94, abs=0.02)      # Cengel: about 10 s
    bad = transient.lumped_capacitance(25, 200, 2100, 8500, 320, 3.0, V, A, t=1)
    assert not bad.valid


@pytest.mark.parametrize("scheme", ["explicit", "implicit"])
def test_plane_wall_fdm_matches_exact_within_1pct(scheme):
    L, alpha, k, h = 0.02, 1e-5, 1.0, 100.0        # Bi = 2, Fo = 1 at t = 40 s
    kw = dict(dt=0.05) if scheme == "implicit" else {}
    x, t, T = transient.plane_wall_fdm(L, alpha, k, h, 100, 20, t_end=40, nx=41,
                                       scheme=scheme, **kw)
    Te = transient.plane_wall_exact(x, 40, L, alpha, k, h, 100, 20)
    err = np.max(np.abs(T[-1] - Te) / (Te - 20))
    assert err < 0.01, err
    assert np.all(np.diff(T[-1]) <= 1e-9)           # hottest at midplane


def test_explicit_stability_check():
    with pytest.raises(ValueError, match="unstable"):
        transient.plane_wall_fdm(0.02, 1e-5, 1.0, 100.0, 100, 20, 10, nx=41, dt=1.0)


def test_exact_solution_one_term_hand():
    # Bi = 1: zeta1 = 0.8603, C1 = 1.1191 (Incropera Table 5.1).
    # At Fo = 0.5, midplane theta* = 1.1191 exp(-0.7401*0.5) = 0.7727
    T = transient.plane_wall_exact(0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
    assert float(T) == pytest.approx(0.7727, abs=5e-4)


# -------------------------------------------------------------- convection
def test_dittus_boelter_hand():
    assert convection.dittus_boelter(1e4, 0.7) == pytest.approx(31.6, abs=0.05)
    with pytest.warns(convection.RangeWarning):
        convection.dittus_boelter(5000, 0.7)


def test_gnielinski_hand():
    assert convection.gnielinski(1e4, 0.7) == pytest.approx(29.8, abs=0.1)
    with pytest.warns(convection.RangeWarning):
        convection.gnielinski(1000, 0.7)


def test_flat_plate_hand():
    assert convection.flat_plate_average(1e5, 0.7) == pytest.approx(186.4, abs=0.2)
    # mixed regime, Re_L = 1e6, Pr = 0.7: (0.037*63096 - 871)*0.8879 = 1299
    assert convection.flat_plate_average(1e6, 0.7) == pytest.approx(1299, rel=0.005)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        convection.flat_plate_average(1e5, 0.7)      # no warning inside range


def test_churchill_chu_hand():
    assert convection.churchill_chu_vertical_plate(1e9, 0.7) == pytest.approx(122.6, abs=0.3)
    with pytest.warns(convection.RangeWarning):
        convection.churchill_chu_horizontal_cylinder(1e13, 0.7)
