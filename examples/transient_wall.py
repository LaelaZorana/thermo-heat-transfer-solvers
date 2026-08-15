"""Plane wall transient: explicit and implicit FDM versus one-term exact solution."""
from _common import FIG
import os
import numpy as np
import matplotlib.pyplot as plt
from thermohx import transient

L, alpha, k, h, Ti, Tinf = 0.02, 1e-5, 1.0, 100.0, 100.0, 20.0
fig, ax = plt.subplots(figsize=(7, 5))
xe, te, Te = transient.plane_wall_fdm(L, alpha, k, h, Ti, Tinf, 60, nx=41, scheme="explicit")
xi, ti, Ti_ = transient.plane_wall_fdm(L, alpha, k, h, Ti, Tinf, 60, nx=41, scheme="implicit", dt=0.1)
for t_plot, c in zip((10, 20, 40, 60), ("C0", "C1", "C2", "C3")):
    ie = int(round(t_plot / (te[1] - te[0]))); ii = int(round(t_plot / (ti[1] - ti[0])))
    ax.plot(xe * 1e3, Te[ie], c, lw=2, label=f"explicit t = {t_plot}s")
    ax.plot(xi * 1e3, Ti_[ii], c, ls="--", lw=1.5)
    xx = np.linspace(0, L, 100)
    ax.plot(xx * 1e3, transient.plane_wall_exact(xx, t_plot, L, alpha, k, h, Ti, Tinf), "k:", lw=1)
ax.plot([], [], "k--", label="implicit"); ax.plot([], [], "k:", label="exact one-term")
ax.set_xlabel("x from midplane [mm]"); ax.set_ylabel("T [C]"); ax.grid(alpha=0.3); ax.legend()
ax.set_title("Plane wall, Bi = 2, explicit vs implicit vs analytical")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "transient_wall.png"), dpi=150)
Tex = transient.plane_wall_exact(xe, 40, L, alpha, k, h, Ti, Tinf)
i40 = int(round(40 / (te[1] - te[0])))
print("max rel error explicit at t=40s:", np.max(np.abs(Te[i40] - Tex) / (Tex - Tinf)))
