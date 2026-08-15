"""Input hygiene: bad inputs must raise a clear ValueError instead of
returning complex numbers, nan, or physically impossible answers."""
import numpy as np
import pytest

from thermohx import cycles, fins, hx, transient, convection

MPa = 1e6
kPa = 1e3


# -------------------------------------------------------------- convection
@pytest.mark.parametrize("fn", [
    lambda: convection.dittus_boelter(-1, 0.7),
    lambda: convection.dittus_boelter(1e4, -0.7),
    lambda: convection.gnielinski(-1, 0.7),
    lambda: convection.gnielinski(500, 0.7),
    lambda: convection.petukhov_friction(-100),
    lambda: convection.churchill_chu_vertical_plate(-1, 0.7),
    lambda: convection.churchill_chu_horizontal_cylinder(-1e5, 0.7),
    lambda: convection.flat_plate_average(-1e5, 0.7),
    lambda: convection.flat_plate_local(-1, 0.7),
])
def test_convection_rejects_nonpositive_inputs(fn):
    with pytest.raises(ValueError):
        fn()


def test_convection_never_returns_complex_or_negative():
    for Re in (5e3, 1e4, 1e6):
        for f in (convection.dittus_boelter, convection.gnielinski,
                  convection.flat_plate_average, convection.flat_plate_local):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Nu = f(Re, 0.7)
            assert np.isreal(Nu) and Nu > 0, (f.__name__, Re)


def test_flat_plate_local_warns_out_of_range():
    with pytest.warns(convection.RangeWarning):
        convection.flat_plate_local(1e4, 100.0)
    with pytest.warns(convection.RangeWarning):
        convection.petukhov_friction(1e7)
    with pytest.warns(convection.RangeWarning):
        convection.churchill_chu_horizontal_cylinder(1e-6, 0.7)


# ------------------------------------------------------------------ cycles
def test_rankine_rejects_compressed_liquid_turbine_inlet():
    with pytest.raises(ValueError, match="liquid or a two phase"):
        cycles.rankine(3 * MPa, 400, 75 * kPa)


def test_rankine_rejects_bad_pressures_and_etas():
    with pytest.raises(ValueError):
        cycles.rankine(75 * kPa, 623.15, 3 * MPa)      # condenser above boiler
    with pytest.raises(ValueError):
        cycles.rankine(3 * MPa, 623.15, 75 * kPa, eta_turbine=1.5)
    with pytest.raises(ValueError):
        cycles.rankine(3 * MPa, 623.15, 75 * kPa, eta_pump=0.0)


def test_brayton_rejects_useless_regenerator():
    # rp = 30 at T3 = 800 K leaves the turbine exhaust colder than the
    # compressor exit, so a regenerator cannot transfer any heat.
    with pytest.raises(ValueError, match="regenerator"):
        cycles.brayton(300, 1e5, 30, 800, regenerator_effectiveness=0.8)


def test_brayton_rejects_bad_effectiveness_and_pressure_ratio():
    with pytest.raises(ValueError):
        cycles.brayton(300, 1e5, 8, 1300, regenerator_effectiveness=1.5)
    with pytest.raises(ValueError):
        cycles.brayton(300, 1e5, 0.5, 1300)


def test_vapor_compression_rejects_bad_inputs():
    with pytest.raises(ValueError):
        cycles.vapor_compression(0.8 * MPa, 0.14 * MPa)   # p_evap above p_cond
    with pytest.raises(ValueError):
        cycles.vapor_compression(0.14 * MPa, 0.8 * MPa, superheat_K=-5)
    with pytest.raises(ValueError):
        cycles.vapor_compression(0.14 * MPa, 0.8 * MPa, subcool_K=-2)


# ---------------------------------------------------------------------- hx
def test_hx_effectiveness_rejects_bad_ntu_cr():
    with pytest.raises(ValueError):
        hx.effectiveness(-1.0, 0.5)
    with pytest.raises(ValueError):
        hx.effectiveness(2.0, 1.5)
    assert hx.effectiveness(0.0, 0.5, "shell_tube_1_2") == 0.0


def test_hx_ntu_inverse_rejects_unattainable():
    with pytest.raises(ValueError):
        hx.ntu_from_effectiveness(0.9, 1.0, "parallel")
    with pytest.raises(ValueError):
        hx.ntu_from_effectiveness(1.0, 0.5, "counter")
    with pytest.raises(ValueError):
        hx.ntu_from_effectiveness(0.9, 1.0, "shell_tube_1_2")
    with pytest.raises(ValueError):
        hx.ntu_from_effectiveness(-0.5, 0.5, "counter")


def test_hx_size_rate_reject_bad_inputs():
    with pytest.raises(ValueError):
        hx.size(1000, 1000, 100, 20, U=100, q=-10e3)
    with pytest.raises(ValueError):
        hx.size(-1000, 1000, 100, 20, U=100, q=10e3)
    with pytest.raises(ValueError):
        hx.size(1000, 1000, 20, 100, U=100, q=10e3)
    with pytest.raises(ValueError):
        hx.rate(1000, 1000, 100, 20, UA=-5)


def test_hx_rate_large_ntu_does_not_raise():
    # Rating near the limiting effectiveness must not fail on the F
    # diagnostic; F may come back as nan there.
    r = hx.rate(1000, 1000, 100, 20, 1e6, "cross_unmixed")
    assert r.q > 0
    assert np.isnan(r.F) or r.F > 0


# -------------------------------------------------------------------- fins
def test_fins_reject_degenerate_geometry():
    with pytest.raises(ValueError):
        fins.pin_fin(10, 200, 0.01, L=0)
    with pytest.raises(ValueError):
        fins.pin_fin(10, 200, 0.01, L=-0.1)
    with pytest.raises(ValueError):
        fins.pin_fin(0, 200, 0.01, 0.1)
    with pytest.raises(ValueError):
        fins.pin_fin(10, 0, 0.01, 0.1)
    assert np.isnan(fins.efficiency_curve(-1.0))


# --------------------------------------------------------------- transient
def test_transient_exact_limiting_biot():
    # Bi = 0: insulated wall stays at T_i.
    T = transient.plane_wall_exact(0.0, 100.0, 0.02, 1e-5, 1.0, 0.0, 100, 20,
                                   n_terms=5)
    assert float(T) == pytest.approx(100.0)
    # Bi = inf: prescribed surface temperature, zeta1 = pi/2, C1 = 4/pi.
    T = transient.plane_wall_exact(0.0, 0.5, 1.0, 1.0, 1.0, np.inf, 1.0, 0.0)
    hand = (4 / np.pi) * np.exp(-(np.pi / 2) ** 2 * 0.5)
    assert float(T) == pytest.approx(hand, rel=1e-9)


def test_transient_exact_rejects_bad_domain():
    with pytest.raises(ValueError):
        transient.plane_wall_exact(0.0, -1.0, 0.02, 1e-5, 1.0, 100.0, 100, 20)
    with pytest.raises(ValueError):
        transient.plane_wall_exact(0.05, 10.0, 0.02, 1e-5, 1.0, 100.0, 100, 20)
    with pytest.warns(transient.RangeWarning):
        transient.plane_wall_exact(0.0, 0.1, 0.02, 1e-5, 1.0, 100.0, 100, 20)


def test_lumped_time_to_reach_validation():
    D = 1e-3
    V, A = np.pi * D ** 3 / 6, np.pi * D ** 2
    with pytest.raises(ValueError):
        transient.lumped_time_to_reach(250, 25, 200, 210, 8500, 320, V, A)
    with pytest.raises(ValueError):
        transient.lumped_time_to_reach(100, 25, 25, 210, 8500, 320, V, A)
    with pytest.warns(transient.RangeWarning):
        transient.lumped_time_to_reach(100, 25, 200, 2100, 8500, 320, V, A, k=3.0)


def test_fdm_rejects_bad_grid():
    with pytest.raises(ValueError):
        transient.plane_wall_fdm(0.02, 1e-5, 1.0, 100.0, 100, 20, t_end=0)
    with pytest.raises(ValueError):
        transient.plane_wall_fdm(0.02, 1e-5, 1.0, 100.0, 100, 20, 10, nx=1)
    with pytest.raises(ValueError):
        transient.plane_wall_fdm(0.02, 1e-5, 1.0, 100.0, 100, 20, 10, dt=0)
