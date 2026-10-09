# -*- coding: utf-8 -*-
"""
physics/pulses.py - Генерация и применение нейромедиаторных импульсов.

Чтобы полностью отключить импульсы: в config.py поставить num_pulses = 0.
Форма импульса - гауссова, параметры pulse_sigma_nm и pulse_center_nm из config.
"""

import numpy as np
import config
from solver.geometry import radial_cell_measures


def pulse_increment_profile(r_values, pulse_r_window, pulse_amount, dr, mode=None):
    """
    Вычисляет гауссов профиль добавки к p(r) при применении импульса.

    Добавка нормирована так, чтобы её интеграл = pulse_amount.
    """
    r_min, r_max = pulse_r_window
    mask = (r_values >= r_min) & (r_values <= r_max)
    if pulse_amount <= 0.0 or not np.any(mask):
        return np.zeros_like(r_values)
    sigma = max(float(config.pulse_sigma_nm), 0.5 * float(dr))
    center = float(np.clip(float(config.pulse_center_nm), float(r_min), float(r_max)))
    cell_measures = radial_cell_measures(r_values, dr, mode)
    added = np.zeros_like(r_values)
    added[mask] = np.exp(-0.5 * ((r_values[mask] - center) / sigma) ** 2)
    added_integral = np.sum(added * cell_measures)
    if added_integral > 0.0:
        added *= pulse_amount / added_integral
    return added


def apply_pulse(p_values, r_values, pulse_r_window, pulse_amount, dr, mode=None, renormalize=True):
    """
    Применяет импульс к распределению p(r).

    renormalize=True  - интерпретация "вероятность одной частицы": после
        добавления импульса полная масса принудительно возвращается к 1
        (импульс переформирует распределение, а не добавляет вещество).
    renormalize=False - интерпретация "концентрация вещества": импульс
        реально добавляет массу pulse_amount, дальнейшая эволюция массы
        определяется только границей и отбором k(r), без принудительной
        нормировки. См. config.normalization_mode.
    """
    added = pulse_increment_profile(r_values, pulse_r_window, pulse_amount, dr, mode)
    p_values = p_values + added
    if renormalize:
        cell_measures = radial_cell_measures(r_values, dr, mode)
        total = np.sum(p_values * cell_measures)
        if total > 0.0:
            p_values /= total
    return p_values


def pulse_indices_from_times(t_values, pulse_times, time_window=None):
    """Находит индексы в t_values, ближайшие к заданным временам импульсов."""
    t_values = np.asarray(t_values)
    pulse_times = np.asarray(pulse_times)
    if t_values.size == 0 or pulse_times.size == 0:
        return np.array([], dtype=int)
    dt_local = t_values[1] - t_values[0] if t_values.size > 1 else 1.0
    idx = np.searchsorted(t_values, pulse_times, side="left")
    idx = np.clip(idx, 0, t_values.size - 1)
    prev_idx = np.clip(idx - 1, 0, t_values.size - 1)
    choose_prev = np.abs(pulse_times - t_values[prev_idx]) <= np.abs(pulse_times - t_values[idx])
    idx = np.where(choose_prev, prev_idx, idx)
    idx = np.unique(idx.astype(int))
    if time_window is not None and idx.size:
        t_start, t_end = time_window
        times_at_idx = t_values[idx]
        tol = 0.5 * dt_local + config.TIME_TOL
        idx = idx[(times_at_idx >= t_start - tol) & (times_at_idx <= t_end + tol)]
    return idx


def compute_pulse_schedule(
    t_values,
    num_pulses,
    pulse_segments=None,
    pulse_time_window=None,
    segment_default_mode="count",
):
    """
    Строит расписание импульсов - список индексов в t_values.

    Поддерживает:
    - равномерное распределение (num_pulses штук в pulse_time_window)
    - сегменты с заданной частотой (Hz) или количеством

    Возвращает: (pulse_indices, segment_info)
    """
    max_freq_per_us = max(float(config.pulse_max_frequency_hz), 0.0) / 1.0e6

    def _event_rate(count_local, t_start_local, t_end_local):
        duration = max(float(t_end_local) - float(t_start_local), 0.0)
        if count_local <= 0 or duration <= 0.0:
            return 0.0
        return float(count_local) / duration

    def _repetition_rate(count_local, t_start_local, t_end_local):
        duration = max(float(t_end_local) - float(t_start_local), 0.0)
        if count_local <= 1 or duration <= 0.0:
            return 0.0
        return float(count_local - 1) / duration

    def _max_count_for_window(t_start_local, t_end_local):
        duration = max(float(t_end_local) - float(t_start_local), 0.0)
        if duration <= 0.0:
            return 1
        if max_freq_per_us <= 0.0:
            return int(1e9)
        return int(np.floor(duration * max_freq_per_us)) + 1

    def _frequency_per_us(value_local, mode_local):
        value_local = max(float(value_local), 0.0)
        mode_local = str(mode_local).lower()
        if mode_local in ("hz", "freq_hz", "frequency_hz"):
            return value_local / 1.0e6
        return value_local

    window_start = None
    window_end = None
    if pulse_time_window is not None:
        window_start = float(min(pulse_time_window))
        window_end = float(max(pulse_time_window))

    if pulse_segments:
        pulse_indices = []
        segment_info = []
        for seg in pulse_segments:
            if isinstance(seg, dict):
                t_start = float(seg.get("t_start", 0.0))
                t_end = float(seg.get("t_end", 0.0))
                mode = str(seg.get("mode", "count")).lower()
                value = float(seg.get("value", seg.get("count", seg.get("frequency", 0.0))))
            else:
                if len(seg) >= 4:
                    t_start, t_end, value, mode = seg[:4]
                    mode = str(mode).lower()
                else:
                    t_start, t_end, value = seg
                    mode = str(segment_default_mode).lower()
            t_start = max(t_values[0], float(t_start))
            t_end = min(t_values[-1], float(t_end))
            if window_start is not None and window_end is not None:
                t_start = max(t_start, window_start)
                t_end = min(t_end, window_end)
            if t_end < t_start:
                continue
            indices = np.array([], dtype=int)
            period = np.inf
            clipped = False
            if mode in ("freq", "frequency", "hz", "freq_hz", "frequency_hz"):
                if value > 0.0:
                    requested_freq = _frequency_per_us(value, mode)
                    freq_capped = min(requested_freq, max_freq_per_us)
                    clipped = freq_capped < requested_freq - 1e-30
                    if freq_capped > 0.0:
                        period = 1.0 / freq_capped
                        pulse_times = np.arange(t_start, t_end + 0.5 * period, period)
                        indices = pulse_indices_from_times(t_values, pulse_times, (t_start, t_end))
            else:
                count = int(round(value))
                if count > 0:
                    count_max = _max_count_for_window(t_start, t_end)
                    if count > count_max:
                        count = count_max
                        clipped = True
                    pulse_times = np.linspace(t_start, t_end, count)
                    indices = pulse_indices_from_times(t_values, pulse_times, (t_start, t_end))
            if indices.size > 1:
                t_segment = t_values[indices]
                period = (t_segment[-1] - t_segment[0]) / (len(indices) - 1)
            freq_hz = (
                _frequency_per_us(value, mode) * 1.0e6
                if mode in ("freq", "frequency", "hz", "freq_hz", "frequency_hz")
                else None
            )
            segment_info.append({
                "t_start": t_start,
                "t_end": t_end,
                "count": int(indices.size),
                "frequency": _event_rate(indices.size, t_start, t_end),
                "repetition_frequency": _repetition_rate(indices.size, t_start, t_end),
                "period": period,
                "mode": mode,
                "clipped": bool(clipped),
                "requested_value": float(value),
                "requested_frequency_hz": freq_hz,
            })
            if indices.size:
                pulse_indices.append(indices)
        if pulse_indices:
            pulse_indices = np.unique(np.concatenate(pulse_indices))
        else:
            pulse_indices = np.array([], dtype=int)
        return pulse_indices, segment_info

    if num_pulses <= 0:
        return np.array([], dtype=int), []

    if pulse_time_window is None:
        t_start, t_end = t_values[0], t_values[-1]
    else:
        window_start, window_end = pulse_time_window
        t_start = float(min(window_start, window_end))
        t_end = float(max(window_start, window_end))
    t_start = max(t_values[0], t_start)
    t_end = min(t_values[-1], t_end)
    if t_end < t_start:
        return np.array([], dtype=int), []

    requested_count = int(round(num_pulses))
    count_max = _max_count_for_window(t_start, t_end)
    count_used = min(max(requested_count, 0), count_max)
    if count_used <= 0:
        return np.array([], dtype=int), []

    pulse_times = np.linspace(t_start, t_end, count_used)
    indices = pulse_indices_from_times(t_values, pulse_times, (t_start, t_end))
    period = (
        (t_end - t_start) / (len(indices) - 1)
        if len(indices) > 1 and t_end > t_start
        else np.inf
    )
    segment_info = [{
        "t_start": t_start,
        "t_end": t_end,
        "count": int(len(indices)),
        "frequency": _event_rate(len(indices), t_start, t_end),
        "repetition_frequency": _repetition_rate(len(indices), t_start, t_end),
        "period": period,
        "clipped": bool(count_used < requested_count),
        "requested_count": int(requested_count),
    }]
    return indices, segment_info
