# -*- coding: utf-8 -*-
"""
tests/_physical_helpers.py - общий запуск решателя для тестов физической ветки
(геометрия Barbour, концентрация в молекулах на нм³, баланс массы).

Каждый тест задаёт ВСЕ физические члены явно; глобальные значения config
меняются только на время вызова и возвращаются обратно.
"""
import contextlib
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config  # noqa: E402
config.use_gpu = False
config.realtime_modeling = False

from solver.fp_solver import solve_fp_equation  # noqa: E402
from utils.units import uptake_to_solver_units  # noqa: E402


@contextlib.contextmanager
def cfg(**kw):
    missing = object()
    old = {k: getattr(config, k, missing) for k in kw}
    try:
        for k, v in kw.items():
            setattr(config, k, v)
        yield
    finally:
        for k, v in old.items():
            if v is missing:
                delattr(config, k)
            else:
                setattr(config, k, v)


def run_physical(geometry="barbour_mixed", D_cleft=330.0, D_pm=320.0, Omega_cleft=0.0, Omega_pm=0.0,
                 N=3000.0, sigma=20.0, window=None, a=200.0, b=400.0, h=20.0, alpha=0.2,
                 R=3000.0, dr=2.0, dt=0.5, T=2000.0, boundary="dirichlet",
                 uptake="none", Vmax_cleft_uM=0.0, Vmax_pm_uM=0.0, Km_uM=20.0,
                 scheme="split_implicit", clip=False, frames=200):
    """Один выброс N молекул в t = 0. Возвращает (c, r, t, info)."""
    window = 5.0 * sigma if window is None else window
    with cfg(geometry_mode=geometry, geometry_a_nm=a, geometry_b_nm=b, cleft_radius_nm=a,
             cleft_height_nm=h, extracellular_volume_fraction=alpha, pulse_center_nm=0.0,
             pulse_sigma_nm=sigma, pulse_max_frequency_hz=1e9, max_saved_frames=frames):
        Km_s, Vc_s, Vp_s = uptake_to_solver_units(Km_uM, Vmax_cleft_uM, Vmax_pm_uM, alpha, geometry, h)
        c, r, t, _, _, info = solve_fp_equation(
            R, dt, dr, T, D_cleft, D_pm, config.xi_s, a, b, Omega_cleft, Omega_pm,
            N, (0.0, window), 1, None, (0.0, 0.0), "count", "schedule", 0.0, 0.0, 0.0, False,
            config.D_transition_kind, config.D_transition_steepness,
            config.Omega_transition_kind, config.Omega_transition_steepness,
            outer_boundary_mode=boundary, normalization_mode="concentration",
            uptake_kind=uptake, Vmax_cleft=Vc_s, Vmax_pm=Vp_s, Km=Km_s,
            uptake_scheme=scheme, clip_negative=clip, track_mass=True,
        )
    assert info["count"] == 1, f"ожидался 1 выброс, получено {info['count']}"
    return c, r, t, info
