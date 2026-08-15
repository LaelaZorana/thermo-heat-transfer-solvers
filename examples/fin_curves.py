"""Fin efficiency versus mL for adiabatic and convective tips, plus a pin fin sweep."""
from _common import FIG
import os
import numpy as np
import matplotlib.pyplot as plt
from thermohx import fins

mL = np.linspace(0.01, 5, 200)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].plot(mL, fins.efficiency_curve(mL), label="adiabatic tip: tanh(mL)/mL")
ax[0].set_xlabel("mL"); ax[0].set_ylabel("fin efficiency"); ax[0].grid(alpha=0.3); ax[0].legend()
L = np.linspace(0.005, 0.2, 100)
for D in (0.003, 0.006, 0.012):
    eff = [fins.pin_fin(50, 200, D, l, tip="convective").effectiveness for l in L]
    eta = [fins.pin_fin(50, 200, D, l, tip="convective").efficiency for l in L]
    ax[1].plot(L * 1e3, eta, label=f"eta, D = {D*1e3:.0f} mm")
ax[1].set_xlabel("pin fin length [mm]"); ax[1].set_ylabel("efficiency (convective tip)")
ax[1].set_title("aluminum pin fins, h = 50 W/m2K"); ax[1].grid(alpha=0.3); ax[1].legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fin_efficiency.png"), dpi=150)
