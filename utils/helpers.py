# -*- coding: utf-8 -*-
"""
utils/helpers.py - Мелкие вспомогательные функции без зависимостей.
"""


def _sanitize_window(window, min_val, max_val):
    """Обрезает окно [start, end] до диапазона [min_val, max_val]."""
    if window is None:
        return float(min_val), float(max_val)
    start, end = window
    start = max(float(min_val), float(start))
    end = min(float(max_val), float(end))
    if end < start:
        start, end = end, start
    return start, end


def _append_radius_unique(radii, value, tol):
    """Добавляет значение в список, если нет дублёра в пределах tol."""
    for existing in radii:
        if abs(existing - value) <= tol:
            return radii
    radii.append(value)
    return radii


def _append_time_unique(times, value, tol):
    """Добавляет момент времени в список, если нет дублёра в пределах tol."""
    for existing in times:
        if abs(existing - value) <= tol:
            return times
    times.append(value)
    return times
