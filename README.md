# thermohx: validated thermodynamics and heat transfer solvers

A small, tested Python library covering the core of an undergraduate thermo and
heat transfer sequence: power and refrigeration cycles on real fluid data
(CoolProp), extended surfaces, heat exchanger sizing and rating, transient
conduction (lumped and finite difference against the exact series), and
convection correlations with validity warnings. Every function carries a
textbook validation case in its docstring, and the test suite checks those
numbers.

![Rankine T-s](figures/rankine_ts.png)

## API overview

| Module | Functions | Notes |
|---|---|---|
| `thermohx.cycles` | `rankine`, `rankine_reheat`, `rankine_regenerative_open_fwh`, `brayton`, `vapor_compression`, `saturation_dome` | CoolProp properties, isentropic efficiencies for turbine, pump, compressor; regenerator effectiveness for Brayton; superheat and subcooling for the refrigeration cycle. Returns eta_th, back-work ratio, specific work, COP, and the state points. |
| `thermohx.fins` | `rectangular_fin`, `pin_fin`, `efficiency_curve` | Adiabatic tip, exact convective tip, and corrected-length tip. Returns efficiency, effectiveness, heat rate, m, mL. |
| `thermohx.hx` | `lmtd`, `correction_factor_F`, `effectiveness`, `ntu_from_effectiveness`, `size`, `rate` | Parallel, counterflow, shell and tube 1-2, crossflow both unmixed. `size` returns area from a duty; `rate` returns outlets from UA. |
| `thermohx.transient` | `lumped_capacitance`, `lumped_time_to_reach`, `plane_wall_exact`, `plane_wall_fdm` | Biot check on lumped results; explicit FDM raises on Fo(1+Bi) > 0.5; implicit uses a banded solve. |
| `thermohx.convection` | `dittus_boelter`, `gnielinski`, `petukhov_friction`, `churchill_chu_vertical_plate`, `churchill_chu_horizontal_cylinder`, `flat_plate_average`, `flat_plate_local` | Each emits a `RangeWarning` outside its fitted range. |

Units are SI throughout (Pa, K, J/kg, W/m2 K). Cycle temperatures are in
kelvin; heat exchanger and transient functions accept any consistent
temperature scale.

```python
from thermohx import cycles, hx
r = cycles.rankine_reheat(15e6, 873.15, 4e6, 873.15, 10e3)
print(r["eta_th"], r["bwr"], r["x_turbine_exit"])
s = hx.size(Ch=213.1, Cc=835.6, Th_in=100, Tc_in=30, U=38.1, Th_out=60, arrangement="counter")
print(s.area, s.lmtd, s.NTU)
```

## Validation table

Reference values were recomputed by hand from steam, air, and R134a tables
before comparing; the code uses CoolProp so small differences from the tables
are expected.

| Case | Source | Reference | thermohx | Test tolerance |
|---|---|---|---|---|
| Rankine 3 MPa, 350 C, 75 kPa | Cengel Thermo Ex 10-1 | eta = 26.0 percent | 26.02 percent | 1 percent |
| Rankine 15 MPa, 600 C, 10 kPa | Cengel Ex 10-3 | eta = 43.0 percent, w_net = 1452.7 kJ/kg | 43.03 percent, 1452.8 kJ/kg | 1 percent |
| Reheat 15 MPa / 4 MPa, 600 C | Cengel Ex 10-4 | eta = 45.0 percent, x_exit = 0.896 | 44.99 percent, 0.896 | 1 percent |
| Open FWH at 1.2 MPa | Cengel Ex 10-5 | y = 0.2270, eta = 46.3 percent | 0.2271, 46.31 percent | 1 percent |
| Brayton rp = 8, 300 to 1300 K | Cengel Ex 9-5 | eta = 42.6 percent, bwr = 0.403 | 42.56 percent, 0.4025 | 1 percent |
| Brayton eta_c 0.80, eta_t 0.85 | Cengel Ex 9-6 | eta = 26.6 percent | 26.61 percent | 1 percent |
| Brayton with 80 percent regenerator | Cengel Ex 9-7 | eta = 36.9 percent | 36.87 percent | 1 percent |
| R134a 0.14 to 0.8 MPa | Cengel Ex 11-1 | COP = 3.97, q_L = 143.7, w = 36.2 kJ/kg | 3.968, 143.7, 36.2 | 1 percent |
| Rectangular fin, k 180, t 3 mm, L 30 mm, h 25 | hand (Incropera Table 3.4) | eta = 0.9729, q = 87.8 W/m | 0.9729, 87.8 | 0.5 percent |
| Counterflow oil cooler | Incropera Ex 11.1 | LMTD 43.2, A = 5.18 m2, L = 65.9 m | 43.20, 5.18, 65.9 | 0.5 percent |
| Shell and tube F, P 0.146, R 3.92 | Bowman closed form | F = 0.9615 | 0.9615 | 5e-4 |
| Lumped thermocouple, D 1 mm | Cengel HT Ex 4-1 | Bi = 0.001, tau = 2.16 s, t99 = 9.9 s | 0.001, 2.159, 9.94 | 1 percent |
| Plane wall Bi = 2, Fo = 1, explicit FDM | vs one-term exact | max relative error | 0.03 percent | 1 percent |
| Plane wall Bi = 2, Fo = 1, implicit FDM | vs one-term exact | max relative error | 0.08 percent | 1 percent |
| Dittus-Boelter Re 1e4, Pr 0.7 | hand | Nu = 31.6 | 31.61 | 0.05 |
| Gnielinski Re 1e4, Pr 0.7 | hand | Nu = 29.8 | 29.82 | 0.1 |
| Flat plate Re_L 1e5 and 1e6, Pr 0.7 | hand | Nu = 186.4 and 1299 | 186.4, 1299 | 0.5 percent |
| Churchill-Chu Ra 1e9, Pr 0.7 | hand | Nu = 122.6 | 122.6 | 0.3 |

Also checked: LMTD area and NTU area agree to 1e-6 for all four arrangements,
sizing then rating round-trips the outlet temperatures, inverse eps-NTU
relations recover NTU, and the explicit solver raises on an unstable step.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .           # or: pip install numpy scipy matplotlib CoolProp pytest
pytest -q                  # 24 tests
python examples/rankine_ts.py
python examples/fin_curves.py
python examples/ntu_curves.py
python examples/transient_wall.py
```

Figures are written to `figures/`: `rankine_ts.png`, `fin_efficiency.png`,
`ntu_curves.png`, `transient_wall.png`.

## Layout

```
src/thermohx/   cycles.py fins.py hx.py transient.py convection.py
examples/       one script per figure
figures/        generated plots
tests/          test_all.py
```
