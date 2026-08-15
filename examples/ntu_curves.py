"""Effectiveness versus NTU for the four arrangements at several Cr."""
from _common import FIG
import os
import numpy as np
import matplotlib.pyplot as plt
from thermohx import hx

NTU = np.linspace(0.01, 5, 150)
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
for ax, arr in zip(axes.flat, hx.ARRANGEMENTS):
    for Cr in (0, 0.25, 0.5, 0.75, 1.0):
        ax.plot(NTU, [hx.effectiveness(n, Cr, arr) for n in NTU], label=f"Cr = {Cr}")
    ax.set_title(arr); ax.grid(alpha=0.3)
axes[0, 0].legend(); 
for a in axes[1]: a.set_xlabel("NTU")
for a in axes[:, 0]: a.set_ylabel("effectiveness")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "ntu_curves.png"), dpi=150)
s = hx.size(0.1 * 2131, 0.2 * 4178, 100, 30, U=38.1, Th_out=60)
print("Incropera 11.1: area", s.area, "LMTD", s.lmtd, "NTU", s.NTU)
