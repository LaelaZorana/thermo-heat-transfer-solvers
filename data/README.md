# Data files

None of this is real plant data and it isn't pretending to be.
Everything in this directory is synthetic or representative, so no file
describes a specific commercial product, plant or engine, and nothing
here came out of a proprietary source. What the numbers do have is
internal consistency, because each one was chosen to sit in a physically
plausible range for the equipment class the file names.

## fluids/

Property tables for dry air at 100 kPa and saturated liquid water, with
temperature in kelvin and SI units throughout. The values are consistent
with the appendix tables of a standard heat transfer textbook, in the
style of Incropera Tables A.4 and A.6, and they serve as the CoolProp
free fallback path in `thermohx.properties`, which interpolates them
linearly. The test suite compares every tabulated value against CoolProp
and holds density and specific heat to 1 percent, viscosity and
conductivity to 2.5 percent, and Prandtl number to 3.5 percent. Prandtl
gets the widest band because it's built from three of the others, so it
inherits all of their error at once.

## heat_exchangers/

Three representative heat exchanger specs, so a car radiator as an
unmixed crossflow core, a shell and tube lube oil cooler with one shell
pass and two tube passes, and a brazed plate water to water unit in
counterflow. Each one gives U, area, and per stream fluid, mass flow and
inlet temperature. `thermohx.specs.rate_hx_spec` rates them with specific
heats pulled from the property tables at the stream mean temperature, or
from the cp written into the spec when the tables don't cover the fluid,
which is what happens with oil.

## cycles/

Two representative plant specs, a 500 MW class subcritical steam plant
with single reheat and one open feedwater heater, and a recuperated gas
turbine of about 5 MW. `thermohx.specs.run_cycle_spec` sends each one
through the matching cycle function and converts the stated net power
into a mass flow, which is the number you'd actually size pipe and pumps
around. `examples/cycle_reports.py` writes the reports and the T-s
figures.
