# -*- coding: utf-8 -*-
"""
visualization/plots.py - Статические графики: снимки p(r), профили sigma.

Ключевая функция _map_targets_to_indices используется через ленивые импорты
из analysis/entropy.py, analysis/scenarios.py и export/reports.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import config
from utils.helpers import _sanitize_window
from utils.formatting import _figure_is_open, _show_figure, _wait_for_close


# ---------------------------------------------------------------------------
# Внутренние вспомогательные функции
# ---------------------------------------------------------------------------

def _map_targets_to_indices(values, targets):
    """
    Находит ближайшие индексы в массиве values для каждого значения из targets.
    Дедуплицирует: один индекс появляется только один раз.
    Возвращает (idx_array, actual_values_array).
    """
    values = np.asarray(values)
    idx_list = []
    actual_list = []
    seen = set()
    for target in targets:
        if not np.isfinite(float(target)):
            continue
        idx = int(np.argmin(np.abs(values - float(target))))
        if idx in seen:
            continue
        seen.add(idx)
        idx_list.append(idx)
        actual_list.append(float(values[idx]))
    return np.array(idx_list, dtype=int), np.array(actual_list, dtype=float)


def _select_time_indices(t_values, times, mode="keyframes", time_window=None, stride=1):
    """
    Выбирает временные индексы в зависимости от режима.
    mode="keyframes" - ближайшие к times (через _map_targets_to_indices).
    mode="realtime"  - все шаги внутри окна с заданным stride.
    """
    mode = str(mode).lower()
    stride = max(int(stride), 1)
    if mode == "realtime":
        if time_window is None:
            times_arr = np.asarray(times) if times is not None else np.array([])
            if times_arr.size:
                t_start = float(np.min(times_arr))
                t_end = float(np.max(times_arr))
            else:
                t_start = float(t_values[0])
                t_end = float(t_values[-1])
        else:
            t_start, t_end = time_window
        t_start, t_end = _sanitize_window((t_start, t_end), t_values[0], t_values[-1])
        mask = (t_values >= t_start - config.TIME_TOL) & (t_values <= t_end + config.TIME_TOL)
        idx = np.where(mask)[0][::stride]
        return idx.astype(int), t_values[idx]
    idx_list, actual_times = _map_targets_to_indices(t_values, times)
    return idx_list, actual_times


def _sigma_label(kind):
    """Подпись оси Y для sigma-графика в зависимости от вида."""
    kind = str(kind).lower()
    if kind == "p":
        return "p(r,t), плотность вероятности"
    if kind == "local_entropy":
        return "s(r,t), локальная энтропия"
    if kind == "epr":
        return "сигма(r,t), плотность производства энтропии"
    return f"сигма[{kind}](r,t)"


def _sigma_title(kind):
    """Заголовок sigma-графика в зависимости от вида."""
    kind = str(kind).lower()
    if kind == "epr":
        return "Плотность производства энтропии сигма(r,t)"
    if kind == "local_entropy":
        return "Локальная энтропия s(r,t)"
    if kind == "p":
        return "Плотность вероятности p(r,t)"
    return _sigma_label(kind)


def _finite_percentile(data, percentile, fallback=1.0):
    """Перцентиль данных, игнорируя NaN/Inf. При пустом массиве возвращает fallback."""
    data = np.asarray(data)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return fallback
    return float(np.percentile(finite, percentile))


# ---------------------------------------------------------------------------
# Нормировка
# ---------------------------------------------------------------------------

def compute_probability_sums(p_history, r_values, dr):
    """
    Вычисляет интеграл p по сетке для каждого временного шага.
    Возвращает 1D-массив размером n_times.
    """
    from solver.geometry import radial_cell_measures  # ленивый импорт
    if p_history.size == 0:
        return np.array([], dtype=float)
    cell_m = radial_cell_measures(r_values, dr)
    return np.sum(p_history * cell_m[:, None], axis=0)


def check_normalization(p_history, r_values, dr, tol=1e-6):
    """
    Проверяет нормировку p(r,t).
    Возвращает (ok, min_sum, max_sum, max_deviation).
    """
    sums = compute_probability_sums(p_history, r_values, dr)
    if sums.size == 0:
        return False, None, None, None
    min_sum = float(np.min(sums))
    max_sum = float(np.max(sums))
    max_dev = float(np.max(np.abs(sums - 1.0)))
    return max_dev <= float(tol), min_sum, max_sum, max_dev


# ---------------------------------------------------------------------------
# Интерактивные графики
# ---------------------------------------------------------------------------

def plot_pr_snapshots(r_values, t_values, p_history, times, pause_sec=0.05):
    """
    Последовательно рисует снимки p(r) для выбранных моментов времени.
    Пользователь закрывает окно для выхода.
    """
    idx_list, actual_times = _map_targets_to_indices(t_values, times)
    if idx_list.size == 0:
        return
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    p_max = np.max(p_history) if p_history.size else 1.0
    p_max = max(p_max, 1e-12)
    ax.set_xlabel("Радиус r, нм")
    ax.set_ylabel("Плотность вероятности p(r)")
    ax.set_xlim(r_values[0], r_values[-1])
    ax.set_ylim(0.0, 1.1 * p_max)
    ax.set_title("Снимки p(r) по времени")
    ax.grid(True)
    _show_figure(fig)
    for idx, t_actual in zip(idx_list, actual_times):
        if not _figure_is_open(fig):
            return
        ax.plot(r_values, p_history[:, idx], lw=2, label=f"t = {t_actual:.2f} мкс")
        ax.legend()
        ax.set_title(f"Снимки p(r): t = {t_actual:.2f} мкс")
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(pause_sec)
    _wait_for_close(fig)


def plot_sigma_lines(
    r_values,
    t_values,
    sigma_history,
    times,
    label="сигма(r,t)",
    title=None,
    use_log=False,
    pause_sec=0.05,
):
    """
    Последовательно рисует профили sigma(r) для выбранных моментов времени.
    """
    idx_list, actual_times = _map_targets_to_indices(t_values, times)
    if idx_list.size == 0:
        return
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    sigma_slice = sigma_history[:, idx_list]
    sigma_max = _finite_percentile(sigma_slice, config.sigma_clip_percentile, fallback=1.0)
    sigma_max = max(sigma_max, 1e-12)
    ax.set_xlabel("Радиус r, нм")
    ax.set_ylabel(label)
    ax.set_xlim(r_values[0], r_values[-1])
    ax.set_ylim(0.0, 1.1 * sigma_max)
    ax.set_title(title or "Профили сигма(r) во времени")
    ax.grid(True)
    _show_figure(fig)
    for idx, t_actual in zip(idx_list, actual_times):
        if not _figure_is_open(fig):
            return
        series = sigma_history[:, idx]
        ax.plot(r_values, series, lw=2, label=f"t = {t_actual:.2f} мкс")
        ax.legend()
        ax.set_title(f"{title or 'Профили сигма(r) во времени'}: t = {t_actual:.2f} мкс")
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(pause_sec)
    _wait_for_close(fig)
