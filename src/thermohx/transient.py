"""Transient conduction: lumped capacitance and 1-D plane wall FDM.

Lumped capacitance validation (Cengel Heat Transfer Ex 4-1): thermocouple
junction, D = 1 mm sphere, k = 35, rho = 8500, cp = 320, h = 210.
Lc = D/6 = 1.667e-4 m, Bi = h Lc / k = 0.001, tau = rho cp Lc / h
= 2.159 s, so 99 percent response (theta = 0.01) takes t = 9.94 s
(Cengel reports 10 s).

Plane wall series solution (Incropera 7e, section 5.5): the wall of
half-thickness L, initially T_i, exposed to T_inf with h on both faces.
theta*(x*, Fo) = sum C_n exp(-zeta_n^2 Fo) cos(zeta_n x*), where zeta_n
solves zeta tan zeta = Bi and C_n = 4 sin zeta_n / (2 zeta_n + sin 2 zeta_n).
The one-term truncation is valid for Fo > 0.2 and a RangeWarning is issued
below that. The FDM solvers are compared to this in tests.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq
from scipy.linalg import solve_banded

from .convection import RangeWarning


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
    if h <= 0 or rho <= 0 or cp <= 0 or k <= 0 or volume <= 0 or area <= 0:
        raise ValueError("h, rho, cp, k, volume and area must all be positive")
    Lc = volume / area
    Bi = h * Lc / k
    tau = rho * cp * volume / (h * area)
    t = np.atleast_1d(np.asarray(t, dtype=float))
    T = T_inf + (T_i - T_inf) * np.exp(-t / tau)
    return LumpedResult(Bi, bool(Bi <= bi_limit), tau, T, t)


def lumped_time_to_reach(T_target, T_i, T_inf, h, rho, cp, volume, area,
                         k=None, bi_limit=0.1):
    """Time for a lumped body to reach T_target. The target must lie strictly
    between T_i and T_inf (exclusive of T_inf, which is only reached
    asymptotically). Pass k to get a Biot validity check, which warns with
    RangeWarning when Bi exceeds bi_limit."""
    if h <= 0 or rho <= 0 or cp <= 0 or volume <= 0 or area <= 0:
        raise ValueError("h, rho, cp, volume and area must all be positive")
    if T_i == T_inf:
        raise ValueError("T_i equals T_inf: the body is already in equilibrium")
    theta = (T_target - T_inf) / (T_i - T_inf)
    if not 0 < theta <= 1:
        raise ValueError(f"T_target = {T_target} is not between T_inf and T_i "
                         "(exclusive of T_inf), so it is never reached")
    if k is not None:
        Bi = h * (volume / area) / k
        if Bi > bi_limit:
            warnings.warn(f"lumped model invalid: Bi = {Bi:.3g} > {bi_limit}",
                          RangeWarning, stacklevel=2)
    tau = rho * cp * volume / (h * area)
    return -tau * np.log(theta)


def _wall_eigenvalue(n, Bi):
    """n-th root of zeta tan zeta = Bi in (n pi, n pi + pi/2)."""
    if np.isinf(Bi):
        return n * np.pi + np.pi / 2
    if Bi == 0:
        return n * np.pi
    lo = n * np.pi + 1e-12
    hi = n * np.pi + np.pi / 2 - 1e-12
    f = lambda z: z * np.tan(z) - Bi
    # For very large Bi the sign change sits extremely close to the upper
    # bracket end; nudge the end outward in float steps until it brackets.
    while f(hi) < 0 and hi < n * np.pi + np.pi / 2:
        hi = np.nextafter(hi, np.inf)
    return brentq(f, lo, hi)


def plane_wall_exact(x, t, L, alpha, k, h, T_i, T_inf, n_terms=1):
    """Series solution for a plane wall of half thickness L, x measured from
    the midplane. Returns temperature array with shape broadcast(x, t).
    Handles Bi = 0 (insulated, uniform T_i) and Bi = inf (prescribed surface
    temperature). Warns when the one-term truncation is used below Fo = 0.2."""
    if L <= 0 or alpha <= 0 or k <= 0 or h < 0:
        raise ValueError("L, alpha and k must be positive and h nonnegative")
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    if np.any(t < 0):
        raise ValueError("t must be nonnegative")
    if np.any(np.abs(x) > L):
        raise ValueError("|x| must not exceed the half thickness L")
    Bi = h * L / k
    Fo = alpha * t / L ** 2
    if n_terms == 1 and np.any(Fo < 0.2):
        warnings.warn("one-term solution requested for Fo < 0.2; increase "
                      "n_terms for accuracy", RangeWarning, stacklevel=2)
    if Bi == 0:
        return T_i + np.zeros(np.broadcast(x, t).shape)
    theta = np.zeros(np.broadcast(x, t).shape)
    for n in range(n_terms):
        z = _wall_eigenvalue(n, Bi)
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
    if L <= 0 or alpha <= 0 or k <= 0 or h < 0:
        raise ValueError("L, alpha and k must be positive and h nonnegative")
    if t_end <= 0:
        raise ValueError("t_end must be positive")
    if nx < 3:
        raise ValueError("nx must be at least 3")
    if dt is not None and dt <= 0:
        raise ValueError("dt must be positive")
    x = np.linspace(0, L, nx)
    dx = x[1] - x[0]
    if dt is None:
        dt = 0.4 * dx ** 2 / alpha if scheme == "explicit" else t_end / 200
    Fo = alpha * dt / dx ** 2
    Bi = h * dx / k
    # This check runs on the requested dt; the rounding below can only shrink
    # dt, so a step that passes here stays stable.
    if scheme == "explicit" and Fo * (1 + Bi) > 0.5 + 1e-12:
        raise ValueError(f"explicit scheme unstable: Fo(1+Bi) = {Fo*(1+Bi):.3f} > 0.5, "
                         f"reduce dt below {0.5*dx**2/(alpha*(1+Bi)):.4g} s")
    nt = max(1, int(round(t_end / dt)))
    if nt * dt < t_end - 1e-12 * t_end:
        nt += 1
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
