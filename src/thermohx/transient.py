"""Transient conduction: lumped capacitance and 1-D plane wall FDM.

Lumped capacitance validation (Cengel Heat Transfer Ex 4-1): thermocouple
junction, D = 1 mm sphere, k = 35, rho = 8500, cp = 320, h = 210.
Lc = D/6 = 1.667e-4 m, Bi = h Lc / k = 0.001, tau = rho cp Lc / h
= 2.159 s, so 99 percent response (theta = 0.01) takes t = 9.94 s
(Cengel reports 10 s).

Plane wall exact one-term solution (Incropera eq 5.40): the wall of
half-thickness L, initially T_i, exposed to T_inf with h on both faces.
theta*(x*, Fo) = C1 exp(-zeta1^2 Fo) cos(zeta1 x*), where zeta1 solves
zeta tan zeta = Bi and C1 = 4 sin zeta1 / (2 zeta1 + sin 2 zeta1).
Valid for Fo > 0.2. The FDM solvers are compared to this in tests.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq
from scipy.linalg import solve_banded


@dataclass
class LumpedResult:
    Bi: float
    valid: bool
    tau: float
    T: np.ndarray
    t: np.ndarray


def lumped_capacitance(T_i, T_inf, h, rho, cp, k, volume, area, t, bi_limit=0.1):
    """T(t) = T_inf + (T_i - T_inf) exp(-t/tau), tau = rho cp V/(h A).
    Biot number Bi = h Lc / k with Lc = V/A. valid is False if Bi > bi_limit."""
    Lc = volume / area
    Bi = h * Lc / k
    tau = rho * cp * volume / (h * area)
    t = np.atleast_1d(np.asarray(t, dtype=float))
    T = T_inf + (T_i - T_inf) * np.exp(-t / tau)
    return LumpedResult(Bi, bool(Bi <= bi_limit), tau, T, t)


def lumped_time_to_reach(T_target, T_i, T_inf, h, rho, cp, volume, area):
    tau = rho * cp * volume / (h * area)
    return -tau * np.log((T_target - T_inf) / (T_i - T_inf))


def _zeta1(Bi):
    if Bi == np.inf:
        return np.pi / 2
    f = lambda z: z * np.tan(z) - Bi
    return brentq(f, 1e-9, np.pi / 2 - 1e-9)


def plane_wall_exact(x, t, L, alpha, k, h, T_i, T_inf, n_terms=1):
    """One-term (or n-term) series solution for a plane wall of half
    thickness L, x measured from the midplane. Returns temperature array
    with shape broadcast(x, t)."""
    Bi = h * L / k
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    Fo = alpha * t / L ** 2
    theta = np.zeros(np.broadcast(x, t).shape)
    for n in range(n_terms):
        lo, hi = n * np.pi + 1e-9, n * np.pi + np.pi / 2 - 1e-9
        z = brentq(lambda z: z * np.tan(z) - Bi, lo, hi)
        C = 4 * np.sin(z) / (2 * z + np.sin(2 * z))
        theta = theta + C * np.exp(-z ** 2 * Fo) * np.cos(z * x / L)
    return T_inf + (T_i - T_inf) * theta


def plane_wall_fdm(L, alpha, k, h, T_i, T_inf, t_end, nx=41, dt=None,
                   scheme="explicit"):
    """1-D transient conduction in a symmetric plane wall of half thickness L,
    nodes from midplane (x=0, symmetry) to surface (x=L, convection).
    Explicit scheme requires Fo_mesh (1 + Bi_mesh) <= 0.5 at the surface
    node; a ValueError is raised if violated. Implicit is unconditionally
    stable. Returns (x, t, T[nt, nx])."""
    x = np.linspace(0, L, nx)
    dx = x[1] - x[0]
    if dt is None:
        dt = 0.4 * dx ** 2 / alpha if scheme == "explicit" else t_end / 200
    Fo = alpha * dt / dx ** 2
    Bi = h * dx / k
    if scheme == "explicit" and Fo * (1 + Bi) > 0.5 + 1e-12:
        raise ValueError(f"explicit scheme unstable: Fo(1+Bi) = {Fo*(1+Bi):.3f} > 0.5, "
                         f"reduce dt below {0.5*dx**2/(alpha*(1+Bi)):.4g} s")
    nt = int(np.ceil(t_end / dt))
    dt = t_end / nt
    Fo = alpha * dt / dx ** 2
    T = np.empty((nt + 1, nx))
    T[0] = T_i
    if scheme == "explicit":
        for n in range(nt):
            Tn = T[n]
            Tnew = Tn.copy()
            Tnew[1:-1] = Tn[1:-1] + Fo * (Tn[2:] - 2 * Tn[1:-1] + Tn[:-2])
            Tnew[0] = Tn[0] + 2 * Fo * (Tn[1] - Tn[0])
            Tnew[-1] = Tn[-1] + 2 * Fo * (Tn[-2] - Tn[-1] + Bi * (T_inf - Tn[-1]))
            T[n + 1] = Tnew
    elif scheme == "implicit":
        # banded matrix (upper, diag, lower)
        ab = np.zeros((3, nx))
        ab[1, :] = 1 + 2 * Fo
        ab[0, 1:] = -Fo           # upper diagonal
        ab[2, :-1] = -Fo          # lower diagonal
        ab[0, 1] = -2 * Fo        # symmetry node
        ab[1, -1] = 1 + 2 * Fo + 2 * Fo * Bi
        ab[2, -2] = -2 * Fo       # surface node
        for n in range(nt):
            rhs = T[n].copy()
            rhs[-1] += 2 * Fo * Bi * T_inf
            T[n + 1] = solve_banded((1, 1), ab, rhs)
    else:
        raise ValueError("scheme must be explicit or implicit")
    return x, np.linspace(0, t_end, nt + 1), T
