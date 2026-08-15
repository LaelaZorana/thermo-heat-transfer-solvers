# Data files

Everything in this directory is synthetic or representative. None of it
describes a specific commercial product, plant or engine, and none of it
comes from a proprietary source. The numbers are internally consistent
and were chosen to be physically plausible for the equipment class each
file names.

## fluids/

Property tables for dry air at 100 kPa and saturated liquid water, with
temperature in kelvin and SI units throughout. The values are consistent
with the appendix tables of a standard heat transfer textbook, in the
style of Incropera Tables A.4 and A.6. They serve as the CoolProp free
fallback path in `thermohx.properties`, which interpolates them linearly.
The test suite compares every tabulated value against CoolProp and holds
density and specific heat to 1 percent, viscosity and conductivity to
2.5 percent, and Prandtl number to 3.5 percent.

## heat_exchangers/

Three representative heat exchanger specs: a car radiator as an unmixed
crossflow core, a shell and tube lube oil cooler with one shell pass and
two tube passes, and a brazed plate water to water unit in counterflow.
Each gives U, area, and per stream fluid, mass flow and inlet temperature.
`thermohx.specs.rate_hx_spec` rates them with specific heats taken from
the property tables at the stream mean temperature, or from the cp given
in the spec for fluids the tables do not cover, such as oil.

## cycles/

Two representative plant specs: a 500 MW class subcritical steam plant
with single reheat and one open feedwater heater, and a recuperated gas
turbine of about 5 MW. `thermohx.specs.run_cycle_spec` runs each through
the matching cycle function and converts the stated net power to a mass
flow. `examples/cycle_reports.py` writes the reports and T-s figures.
