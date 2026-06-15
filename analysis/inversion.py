# -*- coding: utf-8 -*-
"""
analysis/inversion.py - Обнаружение инверсии профиля p(r,t).

Инверсия - это состояние, при котором p(r,t) вне синапса превышает
значение p внутри, что указывает на накопление нейромедиатора снаружи.

Три метода:
- find_peak_inversion_time       - пик p смещается за split_radius
- find_profile_inversion_time    - отношение max(p) во внешней / внутренней зоне
- find_nonmonotonic_inversion_time - профиль становится немонотонным (несколько пиков)
"""

import numpy as np
import config


def find_peak_inversion_time(
    r_values,
    t_values,
    p_history,
    split_radius,
    boundary_guard_nm=0.0,
):
    """
    Возвращает (time_us, peak_radius_nm) в момент, когда пик p(r)
    впервые оказывается снаружи синапса (r > split_radius).
    """
    if split_radius is None or p_history.size == 0:
        return None
    peak_idx = np.argmax(p_history, axis=0)
    peak_r = r_values[peak_idx]
    max_allowed_r = float(r_values[-1]) - max(float(boundary_guard_nm), 0.0)
    mask = (peak_r >= split_radius) & (peak_r <= max_allowed_r + config.TIME_TOL)
    if not np.any(mask):
        return None
    first_idx = int(np.argmax(mask))
    return float(t_values[first_idx]), float(peak_r[first_idx])


def find_profile_inversion_time(
    r_values,
    t_values,
    p_history,
    synapse_radius,
    transition_radius,
    min_ratio=1.0,
    pick="max",
    peak_floor_fraction=0.0,
    boundary_guard_nm=0.0,
):
    """
    Возвращает (time_us, ratio, max_tr, max_syn) в момент наибольшего
    отношения max(p в переходной зоне) / max(p внутри синапса).
    """
    if p_history.size == 0:
        return None
    r_values = np.asarray(r_values)
    syn_mask = r_values <= float(synapse_radius)
    upper_bound = float(r_values[-1]) - max(float(boundary_guard_nm), 0.0)
    tr_mask = (
        (r_values >= float(synapse_radius))
        & (r_values <= float(transition_radius))
        & (r_values <= upper_bound + config.TIME_TOL)
    )
    if not np.any(syn_mask) or not np.any(tr_mask):
        return None
    max_syn = np.max(p_history[syn_mask, :], axis=0)
    max_tr = np.max(p_history[tr_mask, :], axis=0)
    ratio = max_tr / np.maximum(max_syn, 1e-30)
    initial_peak = max(float(np.max(p_history[:, 0])), 1e-30)
    floor_value = max(float(peak_floor_fraction), 0.0) * initial_peak
    valid = max_tr >= floor_value
    pick = str(pick).lower()
    if pick == "first":
        idx_list = np.where(valid & (ratio >= float(min_ratio)))[0]
        if idx_list.size == 0:
            return None
        idx = int(idx_list[0])
    else:
        scored_ratio = np.where(valid, ratio, -np.inf)
        idx = int(np.argmax(scored_ratio))
        if not np.isfinite(scored_ratio[idx]) or ratio[idx] < float(min_ratio):
            return None
    return (
        float(t_values[idx]),
        float(ratio[idx]),
        float(max_tr[idx]),
        float(max_syn[idx]),
    )


def find_nonmonotonic_inversion_time(
    r_values,
    t_values,
    p_history,
    peak_floor_fraction=1e-4,
    boundary_guard_nm=0.0,
):
    """
    Возвращает dict с временем первого появления немонотонного профиля
    (несколько пиков) и координатами пиков/впадин.
    """
    if p_history.size == 0:
        return None
    r_values = np.asarray(r_values, dtype=float)
    upper_bound = float(r_values[-1]) - max(float(boundary_guard_nm), 0.0)
    valid_mask = r_values <= upper_bound + config.TIME_TOL
    if np.count_nonzero(valid_mask) < 5:
        valid_mask = np.ones_like(r_values, dtype=bool)
    r_valid = r_values[valid_mask]
    profiles = p_history[valid_mask, :]
    peak_floor = max(float(peak_floor_fraction), 0.0) * max(float(np.max(p_history[:, 0])), 1e-30)
    slc = profiles[:, 1:]
    interior = slc[1:-1]
    is_peak = (interior > slc[:-2]) & (interior > slc[2:])
    is_trough = (interior < slc[:-2]) & (interior < slc[2:])
    peaks_count = np.sum(is_peak, axis=0)
    troughs_count = np.sum(is_trough, axis=0)
    del is_peak, is_trough
    max_vals = np.max(slc, axis=0)
    candidates = np.where(
        (max_vals >= peak_floor) & (peaks_count >= 2) & (troughs_count >= 1)
    )[0]
    if candidates.size == 0:
        return None
    t_local = int(candidates[0])
    t_idx = t_local + 1
    profile = profiles[:, t_idx]
    peaks = np.where((profile[1:-1] > profile[:-2]) & (profile[1:-1] > profile[2:]))[0] + 1
    troughs = np.where((profile[1:-1] < profile[:-2]) & (profile[1:-1] < profile[2:]))[0] + 1
    return {
        "time": float(t_values[t_idx]),
        "profile_index": int(t_idx),
        "peak_radii": r_valid[peaks],
        "trough_radii": r_valid[troughs],
    }
