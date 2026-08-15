"""Run the plant specs in data/cycles/, print a report for each, and draw
a T-s figure for each: the steam plant on the water dome, the gas turbine
as its state point path."""
from _common import FIG
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from CoolProp.CoolProp import PropsSI
from thermohx import cycles, specs

report_lines = []


def report(spec, r):
    lines = [spec["name"]]
    lines.append(f"  eta_th = {r['eta_th']*100:.2f} percent, bwr = {r['bwr']:.3f}")
    lines.append(f"  w_net = {r['w_net']/1e3:.1f} kJ/kg, q_in = {r['q_in']/1e3:.1f} kJ/kg")
    if "y" in r:
        lines.append(f"  bleed fraction y = {r['y']:.4f}, "
                     f"turbine exit quality = {r['x_turbine_exit']:.3f}")
    if "m_dot_kg_s" in r:
        lines.append(f"  net power = {r['net_power_W']/1e6:.0f} MW, "
                     f"m_dot = {r['m_dot_kg_s']:.1f} kg/s, "
                     f"heat input = {r['heat_input_W']/1e6:.0f} MW")
    return "\n".join(lines)


def isobar(p, s_from, s_to, fluid, n=80):
    s = np.linspace(s_from, s_to, n)
    T = [PropsSI("T", "P", p, "S", si, fluid) for si in s]
    return s / 1e3, np.array(T) - 273.15


# ---------------------------------------------------------- steam plant
spec = specs.load_spec(specs.DATA_DIR / "cycles" / "steam_plant_500mw.yaml")
r = specs.run_cycle_spec(spec)
report_lines.append(report(spec, r))
st = {s.name: s for s in r["states"]}
inp = {k: float(v) for k, v in spec["inputs"].items()}
sl, Tl, sv, Tv = cycles.saturation_dome("Water")
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(sl / 1e3, Tl - 273.15, "k", lw=1)
ax.plot(sv / 1e3, Tv - 273.15, "k", lw=1)
ax.plot(*isobar(inp["p_boiler"], st["4"].s, st["5"].s, "Water"), "r", lw=2, label="boiler")
ax.plot([st["5"].s / 1e3, st["6"].s / 1e3], [st["5"].T - 273.15, st["6"].T - 273.15], "b", lw=2)
ax.plot(*isobar(inp["p_reheat"], st["6"].s, st["7"].s, "Water"), "orange", lw=2, label="reheat")
ax.plot([st["7"].s / 1e3, st["8"].s / 1e3, st["9"].s / 1e3],
        [st["7"].T - 273.15, st["8"].T - 273.15, st["9"].T - 273.15], "b", lw=2, label="turbines")
ax.plot(*isobar(inp["p_condenser"], st["9"].s, st["1"].s, "Water"), "g", lw=2, label="condenser")
for name, s in st.items():
    dx, dy = (-13, -3) if name in ("2", "4") else (5, 5)
    ax.annotate(name, (s.s / 1e3, s.T - 273.15), textcoords="offset points", xytext=(dx, dy))
ax.set_xlabel("s [kJ/kg K]"); ax.set_ylabel("T [C]")
ax.set_title(f"500 MW class steam plant, eta_th = {r['eta_th']*100:.1f} percent")
ax.legend(loc="upper left"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "cycle_steam_ts.png"), dpi=150)

# ----------------------------------------------------------- gas turbine
spec = specs.load_spec(specs.DATA_DIR / "cycles" / "gas_turbine_5mw_recuperated.yaml")
r = specs.run_cycle_spec(spec)
report_lines.append(report(spec, r))
st = {s.name: s for s in r["states"]}
fig, ax = plt.subplots(figsize=(7, 5))
order = ["1", "2", "5", "3", "4"]  # include recuperator exit 5 on the cold side
pts = [st[n] for n in order if n in st]
ax.plot([s.s / 1e3 for s in pts], [s.T for s in pts], "o-", lw=2, label="cycle path")
for s in pts:
    ax.annotate(s.name, (s.s / 1e3, s.T), textcoords="offset points", xytext=(6, 4))
ax.set_xlabel("s [kJ/kg K]"); ax.set_ylabel("T [K]")
ax.set_title(f"Recuperated 5 MW gas turbine, eta_th = {r['eta_th']*100:.1f} percent")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "cycle_gasturbine_ts.png"), dpi=150)

text = "\n\n".join(report_lines) + "\n"
print(text)
out = Path(FIG) / "cycle_reports.txt"
out.write_text(text)
print("wrote", out)
