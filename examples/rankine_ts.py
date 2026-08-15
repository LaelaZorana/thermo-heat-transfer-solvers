"""T-s diagram of the ideal reheat Rankine cycle (Cengel Ex 10-4 conditions)."""
from _common import FIG
import os
import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI
from thermohx import cycles

MPa = 1e6
r = cycles.rankine_reheat(15 * MPa, 873.15, 4 * MPa, 873.15, 10e3)
sl, Tl, sv, Tv = cycles.saturation_dome("Water")
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(sl / 1e3, Tl - 273.15, "k", lw=1)
ax.plot(sv / 1e3, Tv - 273.15, "k", lw=1)

def isobar(p, s_from, s_to, n=80):
    s = np.linspace(s_from, s_to, n)
    T = [PropsSI("T", "P", p, "S", si, "Water") for si in s]
    return s / 1e3, np.array(T) - 273.15

st = {s.name: s for s in r["states"]}
# 1-2 pump (tiny), 2-3 boiler at 15 MPa, 3-4 turbine, 4-5 reheat at 4 MPa, 5-6 turbine, 6-1 condenser
ax.plot(*isobar(15 * MPa, st["2"].s, st["3"].s), "r", lw=2, label="boiler 15 MPa")
ax.plot([st["3"].s / 1e3, st["4"].s / 1e3], [st["3"].T - 273.15, st["4"].T - 273.15], "b", lw=2)
ax.plot(*isobar(4 * MPa, st["4"].s, st["5"].s), "orange", lw=2, label="reheat 4 MPa")
ax.plot([st["5"].s / 1e3, st["6"].s / 1e3], [st["5"].T - 273.15, st["6"].T - 273.15], "b", lw=2, label="turbines")
ax.plot(*isobar(10e3, st["6"].s, st["1"].s), "g", lw=2, label="condenser 10 kPa")
for k, s in st.items():
    ax.annotate(k, (s.s / 1e3, s.T - 273.15), textcoords="offset points", xytext=(5, 5))
ax.set_xlabel("s [kJ/kg K]"); ax.set_ylabel("T [C]")
ax.set_title(f"Ideal reheat Rankine, eta_th = {r['eta_th']*100:.1f} percent, x_exit = {r['x_turbine_exit']:.3f}")
ax.legend(loc="upper left"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "rankine_ts.png"), dpi=150)
print("eta_th", r["eta_th"], "w_net kJ/kg", r["w_net"] / 1e3, "bwr", r["bwr"])
