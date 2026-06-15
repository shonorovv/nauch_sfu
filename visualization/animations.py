# -*- coding: utf-8 -*-
"""
visualization/animations.py - Анимации: p(r,t), p(t), sigma, энтропия.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import config
from utils.helpers import _sanitize_window
from utils.formatting import _figure_is_open, _show_figure, _wait_for_close
from visualization.plots import (
    _map_targets_to_indices,
    _select_time_indices,
    _finite_percentile,
)


def animate_pr_slices(
    r_values,
    t_values,
    p_history,
    times,
    pause_sec=0.05,
    mode="keyframes",
    time_window=None,
    stride=1,
    trail=False,
    trail_alpha=0.2,
    trail_max=None,
):
    """
    Анимирует профиль p(r) во времени.
    mode="keyframes" - переходит между ключевыми кадрами.
    mode="realtime"  - пробегает все шаги внутри time_window с заданным stride.
    trail=True добавляет след из предыдущих профилей.
    """
    idx_list, actual_times = _select_time_indices(
        t_values, times, mode=mode, time_window=time_window, stride=stride
    )
    if idx_list.size == 0:
        return
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    p_max = np.max(p_history) if p_history.size else 1.0
    p_max = max(p_max, 1e-12)
    line, = ax.plot([], [], lw=2.5)
    trail_lines = []
    ax.set_xlabel("Радиус r, нм")
    ax.set_ylabel("Плотность вероятности p(r)")
    ax.set_xlim(r_values[0], r_values[-1])
    ax.set_ylim(0.0, 1.1 * p_max)
    ax.grid(True)
    ax.set_title("Профиль p(r) во времени")
    _show_figure(fig)
    for idx, t_actual in zip(idx_list, actual_times):
        if not _figure_is_open(fig):
            return
        if trail:
            trail_line, = ax.plot(
                r_values,
                p_history[:, idx],
                lw=1.2,
                alpha=float(trail_alpha),
                color=line.get_color(),
            )
            trail_lines.append(trail_line)
            if trail_max is not None and len(trail_lines) > int(trail_max):
                old = trail_lines.pop(0)
                old.remove()
        line.set_data(r_values, p_history[:, idx])
        ax.set_title(f"Профиль p(r) при t = {t_actual:.2f} мкс")
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(pause_sec)
    _wait_for_close(fig)


def animate_pt_series(
    r_values,
    t_values,
    p_history,
    radii,
    pause_sec=0.05,
    pulse_times=None,
    show_pulse_markers=False,
    highlight_radii=None,
):
    """
    Анимирует p(t) при нескольких фиксированных радиусах.
    Опционально отображает вертикальные маркеры импульсов.
    """
    idx_list, actual_radii = _map_targets_to_indices(r_values, radii)
    if idx_list.size == 0 or len(t_values) <= 1:
        return
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    p_max = max(float(np.max(p_history[idx, :])) for idx in idx_list)
    p_max = max(p_max, 1e-12)
    lines_pt = []
    highlight_radii = highlight_radii or []
    highlight_set = set(
        int(i) for i, r_val in zip(idx_list, actual_radii) if r_val in highlight_radii
    )
    for idx, r_actual in zip(idx_list, actual_radii):
        if idx in highlight_set:
            line, = ax.plot([], [], lw=2.5, label=f"r = {r_actual:.2f} нм (начало спада D)")
        else:
            line, = ax.plot([], [], label=f"r = {r_actual:.2f} нм")
        lines_pt.append((line, idx))
    ax.set_xlabel("Время t, мкс")
    ax.set_ylabel("Плотность вероятности p(t)")
    ax.set_xlim(t_values[0], t_values[-1])
    ax.set_ylim(0.0, 1.1 * p_max)
    pulse_arr = np.asarray(pulse_times) if pulse_times is not None else np.array([])
    pulse_lc = None
    if show_pulse_markers and pulse_arr.size:
        pulse_lc = LineCollection([], colors="k", alpha=0.15, linewidths=0.8)
        ax.add_collection(pulse_lc)
    ax.legend()
    ax.grid(True)
    ax.set_title("Эволюция p(t) при фиксированном r")
    _show_figure(fig)
    for t_idx in range(len(t_values)):
        if not _figure_is_open(fig):
            return
        if pulse_lc is not None:
            t_now = t_values[t_idx]
            visible = pulse_arr[pulse_arr <= t_now + config.TIME_TOL]
            segs = [((t, 0.0), (t, 1.1 * p_max)) for t in visible]
            pulse_lc.set_segments(segs)
        t_slice = t_values[0 : t_idx + 1]
        for line, r_idx in lines_pt:
            line.set_data(t_slice, p_history[r_idx, 0 : t_idx + 1])
        ax.set_title(f"p(t) при фиксированном r: t = {t_values[t_idx]:.2f} мкс")
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(pause_sec)
    _wait_for_close(fig)


def animate_sigma_heatmap(
    r_values,
    t_values,
    sigma_history,
    r_window,
    t_window,
    pause_sec=0.05,
    label="сигма(r,t)",
    title=None,
):
    """
    Анимирует тепловую карту sigma(r,t) в виде imshow: строки добавляются по мере счёта.
    """
    if sigma_history.size == 0:
        return
    r_min, r_max_local = _sanitize_window(r_window, r_values[0], r_values[-1])
    t_min, t_max_local = _sanitize_window(t_window, t_values[0], t_values[-1])
    r_mask = (r_values >= r_min) & (r_values <= r_max_local)
    t_mask = (t_values >= t_min) & (t_values <= t_max_local)
    r_sel = r_values[r_mask]
    t_sel = t_values[t_mask]
    if r_sel.size == 0 or t_sel.size == 0:
        return
    sigma_sel = sigma_history[np.ix_(r_mask, t_mask)]
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    sigma_max = _finite_percentile(sigma_sel, config.sigma_clip_percentile, fallback=1.0)
    sigma_max = max(sigma_max, 1e-12)
    heatmap = np.full((t_sel.size, r_sel.size), np.nan)
    cmap = plt.cm.magma.copy()
    cmap.set_bad(color="white", alpha=0.0)
    im = ax.imshow(
        heatmap,
        aspect="auto",
        origin="lower",
        extent=[r_sel[0], r_sel[-1], t_sel[0], t_sel[-1]],
        vmin=0.0,
        vmax=1.1 * sigma_max,
        cmap=cmap,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label)
    ax.set_xlabel("Радиус r, нм")
    ax.set_ylabel("Время t, мкс")
    ax.set_title(title or "Тепловая карта сигма(r,t)")
    _show_figure(fig)
    for t_idx in range(t_sel.size):
        if not _figure_is_open(fig):
            return
        heatmap[t_idx, :] = sigma_sel[:, t_idx]
        im.set_data(heatmap)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(pause_sec)
    _wait_for_close(fig)


def animate_shannon_entropy(t_values, shannon_entropy, pause_sec=0.05):
    """
    Анимирует кривую интегральной энтропии Шеннона S(t).
    """
    if len(t_values) <= 1:
        return
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    s_max = np.max(shannon_entropy) if len(shannon_entropy) > 1 else 1.0
    s_max = max(s_max, 1e-12)
    line, = ax.plot([], [], lw=2)
    ax.set_xlabel("Время t, мкс")
    ax.set_ylabel("Энтропия Шеннона S(t), интеграл по r")
    ax.set_xlim(t_values[0], t_values[-1])
    ax.set_ylim(0.0, 1.1 * s_max)
    ax.grid(True)
    ax.set_title("Энтропия Шеннона S(t)")
    _show_figure(fig)
    for t_idx in range(len(t_values)):
        if not _figure_is_open(fig):
            return
        line.set_data(t_values[0 : t_idx + 1], shannon_entropy[0 : t_idx + 1])
        ax.set_title(f"Энтропия Шеннона S(t): t = {t_values[t_idx]:.2f} мкс")
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(pause_sec)
    _wait_for_close(fig)


def animate_local_entropy_at_radius(
    r_values, t_values, local_entropy, radius, pause_sec=0.05
):
    """
    Анимирует локальную энтропию s(t) при одном или нескольких фиксированных радиусах.
    radius может быть числом или списком чисел.
    """
    radii = [radius] if np.isscalar(radius) else list(radius)
    idx_list, actual_radii = _map_targets_to_indices(r_values, radii)
    if idx_list.size == 0 or len(t_values) <= 1:
        return
    if config.VERBOSE:
        r_info = ", ".join(f"{r:.2f} нм" for r in actual_radii)
        print(f"График локальной энтропии: r = {r_info}")
    series_list = [local_entropy[int(idx), :] for idx in idx_list]
    s_max = max(float(np.max(series)) for series in series_list)
    s_max = max(s_max, 1e-12)
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    lines = []
    for r_actual in actual_radii:
        line, = ax.plot([], [], lw=2, label=f"r = {r_actual:.2f} нм")
        lines.append(line)
    ax.set_xlabel("Время t, мкс")
    ax.set_ylabel("Локальная энтропия s(t) при фиксированном r")
    ax.set_xlim(t_values[0], t_values[-1])
    ax.set_ylim(0.0, 1.1 * s_max)
    ax.grid(True)
    ax.legend()
    ax.set_title("Локальная энтропия s(t) при фиксированном r")
    _show_figure(fig)
    for t_idx in range(len(t_values)):
        if not _figure_is_open(fig):
            return
        t_slice = t_values[0 : t_idx + 1]
        for line, series in zip(lines, series_list):
            line.set_data(t_slice, series[0 : t_idx + 1])
        ax.set_title(f"Локальная энтропия s(t): t = {t_values[t_idx]:.2f} мкс")
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(pause_sec)
    _wait_for_close(fig)
