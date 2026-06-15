# -*- coding: utf-8 -*-
"""
analysis/entropy.py - Энтропия, скорость производства энтропии (EPR), оболочечные вероятности.
"""

import numpy as np
import config
from solver.geometry import radial_cell_measures


def compute_entropy_metrics(p_history, r_values, dr):
    """Локальная и шенноновская энтропия распределения p(r,t)."""
    eps = 1e-30
    p_safe = np.maximum(p_history, eps)
    local_entropy = -p_safe * np.log(p_safe)
    cell_measures = radial_cell_measures(r_values, dr)
    shannon_entropy = np.sum(local_entropy * cell_measures[:, None], axis=0)
    return local_entropy, shannon_entropy


def compute_sigma_history(
    p_history,
    D_values,
    Omega_values,
    dr,
    kind="epr",
    local_entropy=None,
    p_min=None,
    d_min=None,
    r_values=None,
):
    """
    Вычисляет поле sigma(r,t) для отображения на тепловых картах.

    kind:
    - "epr"           - скорость производства энтропии (flux^2 / D*p)
    - "p"             - само распределение p(r,t)
    - "local_entropy" - локальная энтропийная плотность
    """
    kind = str(kind).lower()
    if kind == "p":
        return p_history
    if kind == "local_entropy":
        if local_entropy is not None:
            return local_entropy
        if r_values is None:
            r_values = (np.arange(p_history.shape[0]) + 0.5) * dr
        local_entropy, _ = compute_entropy_metrics(p_history, r_values, dr)
        return local_entropy
    if kind != "epr":
        raise ValueError(f"Неизвестный тип сигма: {kind}")
    if p_history.size == 0:
        return np.zeros_like(p_history)
    p_min = 0.0 if p_min is None else float(p_min)
    d_min = 0.0 if d_min is None else float(d_min)
    dp_dr = np.empty_like(p_history)
    dp_dr[1:-1] = (p_history[2:] - p_history[:-2]) / (2.0 * dr)
    dp_dr[0]    = (p_history[1]  - p_history[0])  / dr
    dp_dr[-1]   = (p_history[-1] - p_history[-2]) / dr
    flux = -D_values[:, None] * dp_dr + Omega_values[:, None] * p_history
    if p_min > 0.0 or d_min > 0.0:
        mask = (p_history > p_min) & (D_values[:, None] > d_min)
        safe = np.where(mask, D_values[:, None] * p_history, 1.0)
        sigma = np.where(mask, flux * flux / safe, np.nan)
    else:
        denom = np.maximum(D_values[:, None] * p_history, 1e-30)
        sigma = flux * flux / denom
    return sigma


def _shell_measures(r_values, dr):
    return radial_cell_measures(r_values, dr)


def compute_shell_probabilities(p_history, r_values, dr):
    """q(r,t) = p(r,t) * dV(r) - вероятность обнаружить частицу в оболочке."""
    measures = _shell_measures(r_values, dr)[:, None]
    return np.asarray(p_history, dtype=float) * measures


def compute_local_entropy_fields(p_history, r_values, dr):
    """Возвращает словарь с локальными полями энтропии."""
    p_safe = np.maximum(np.asarray(p_history, dtype=float), config.EPS)
    shell_probs = np.maximum(compute_shell_probabilities(p_history, r_values, dr), config.EPS)
    local_entropy_density = -p_safe * np.log(p_safe)
    local_shell_entropy = -shell_probs * np.log(shell_probs)
    measures = _shell_measures(r_values, dr)[:, None]
    shannon_continuous = np.sum(local_entropy_density * measures, axis=0)
    shannon_shell = np.sum(local_shell_entropy, axis=0)
    return {
        "local_entropy_density": local_entropy_density,
        "local_shell_entropy": local_shell_entropy,
        "shell_probabilities": shell_probs,
        "shannon_continuous": shannon_continuous,
        "shannon_shell": shannon_shell,
    }


def analyze_local_entropy_consistency(
    r_values,
    t_values,
    p_history,
    local_entropy_density,
    local_shell_entropy,
    target_times,
):
    """Проверяет соответствие между пиком p и пиком энтропии в ключевые моменты."""
    from visualization.plots import _map_targets_to_indices
    idx_list, actual_times = _map_targets_to_indices(t_values, target_times)
    checks = []
    for idx, actual_time in zip(idx_list, actual_times):
        profile = p_history[:, idx]
        density_entropy = local_entropy_density[:, idx]
        shell_entropy = local_shell_entropy[:, idx]
        peak_idx = int(np.argmax(profile))
        density_peak_idx = int(np.argmax(density_entropy))
        shell_peak_idx = int(np.argmax(shell_entropy))
        peak_p = float(profile[peak_idx])
        checks.append({
            "time": float(actual_time),
            "peak_radius": float(r_values[peak_idx]),
            "peak_probability": peak_p,
            "entropy_density_at_peak": float(density_entropy[peak_idx]),
            "entropy_density_peak_radius": float(r_values[density_peak_idx]),
            "entropy_density_peak_value": float(density_entropy[density_peak_idx]),
            "shell_entropy_at_peak": float(shell_entropy[peak_idx]),
            "shell_entropy_peak_radius": float(r_values[shell_peak_idx]),
            "shell_entropy_peak_value": float(shell_entropy[shell_peak_idx]),
            "density_entropy_monotone_theory": bool(peak_p < np.exp(-1.0)),
        })
    return checks


def compare_entropy_at_points(
    r_values,
    t_values,
    p_history,
    local_entropy_density,
    local_shell_entropy,
    shell_probabilities,
    radii,
    times,
):
    """Табличное сравнение энтропии в заданных точках (r, t)."""
    from visualization.plots import _map_targets_to_indices
    r_idx, r_actual = _map_targets_to_indices(r_values, radii)
    t_idx, t_actual = _map_targets_to_indices(t_values, times)
    rows = []
    for ti, t_val in zip(t_idx, t_actual):
        p_slice = p_history[:, ti]
        density_slice = local_entropy_density[:, ti]
        shell_prob_slice = shell_probabilities[:, ti]
        shell_entropy_slice = local_shell_entropy[:, ti]
        p_peak_idx = int(np.argmax(p_slice))
        density_peak_idx = int(np.argmax(density_slice))
        shell_entropy_peak_idx = int(np.argmax(shell_entropy_slice))
        density_monotone = bool(float(np.max(p_slice)) < np.exp(-1.0))
        for ri, r_val in zip(r_idx, r_actual):
            p_val = float(p_slice[ri])
            density_val = float(density_slice[ri])
            shell_prob_val = float(shell_prob_slice[ri])
            shell_entropy_val = float(shell_entropy_slice[ri])
            rows.append({
                "time": float(t_val),
                "radius": float(r_val),
                "p": p_val,
                "local_entropy_density": density_val,
                "shell_probability": shell_prob_val,
                "local_shell_entropy": shell_entropy_val,
                "p_rank": int(1 + np.count_nonzero(p_slice > p_val)),
                "density_entropy_rank": int(1 + np.count_nonzero(density_slice > density_val)),
                "shell_entropy_rank": int(1 + np.count_nonzero(shell_entropy_slice > shell_entropy_val)),
                "p_peak_radius": float(r_values[p_peak_idx]),
                "density_entropy_peak_radius": float(r_values[density_peak_idx]),
                "shell_entropy_peak_radius": float(r_values[shell_entropy_peak_idx]),
                "density_entropy_monotone_region": density_monotone,
            })
    return rows
