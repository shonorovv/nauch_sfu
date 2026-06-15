# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import config

LIVE_MODELING_FIGURES = []


def _can_show_realtime_modeling():
    backend = str(matplotlib.get_backend()).lower()
    return bool(config.realtime_modeling) and "agg" not in backend


class RealtimeModelingPlot:

    def __init__(self, r_values, t_values, p_initial, pulse_mask, pulse_time_window, radii):
        from visualization.plots import _map_targets_to_indices
        self.enabled = _can_show_realtime_modeling()
        if not self.enabled:
            self.fig = None
            return
        self.r_values = r_values
        self.t_max = float(t_values[-1])
        self.t_min = float(t_values[0])
        self.indices, self.actual_radii = _map_targets_to_indices(r_values, radii)
        p0 = np.asarray(p_initial, dtype=float)
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 1, figsize=(11, 9))
        self.profile_line, = self.axes[0].plot(r_values, p0, lw=2)
        self.accumulated_t = [self.t_min]
        self.accumulated_p = {int(idx): [float(p0[idx])] for idx in self.indices}
        self.trace_lines = []
        for idx, r_actual in zip(self.indices, self.actual_radii):
            line, = self.axes[1].plot(
                [self.t_min],
                [float(p0[idx])],
                lw=1.8,
                label=f"r={r_actual:.1f} нм",
            )
            self.trace_lines.append((line, int(idx)))
        pulse_times = t_values[np.asarray(pulse_mask, dtype=bool)]
        if pulse_time_window is not None:
            pulse_start = max(float(t_values[0]), float(min(pulse_time_window)))
            pulse_end = min(float(t_values[-1]), float(max(pulse_time_window)))
            self.axes[1].axvspan(pulse_start, pulse_end, color="tab:red", alpha=0.08)
        for pulse_time in pulse_times:
            self.axes[1].axvline(pulse_time, color="k", alpha=0.12, lw=0.8)
        self.axes[0].set_xlabel("Радиус r, нм")
        self.axes[0].set_ylabel("p(r)")
        self.axes[0].set_xlim(float(r_values[0]), float(r_values[-1]))
        self.axes[0].set_ylim(0.0, max(float(np.max(p0)) * 1.1, 1e-12))
        self.axes[0].set_title("Моделирование в реальном времени: профиль p(r)")
        self.axes[0].grid(True)
        self.axes[1].set_xlabel("Время t, мкс")
        self.axes[1].set_ylabel("p(t)")
        self.axes[1].set_xlim(self.t_min, self.t_max)
        ymax_init = max(float(np.max(p0[self.indices])) * 1.1 if self.indices.size else 0.0, 1e-12)
        self.axes[1].set_ylim(0.0, ymax_init)
        self.axes[1].set_title("Моделирование в реальном времени: отклик по радиусам")
        self.axes[1].grid(True)
        if self.trace_lines:
            self.axes[1].legend()
        self.fig.tight_layout()
        LIVE_MODELING_FIGURES.append(self.fig)

    def update(self, t_current, p_current):
        if not self.enabled:
            return
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            self.enabled = False
            return
        p_arr = np.asarray(p_current, dtype=float)
        self.profile_line.set_data(self.r_values, p_arr)
        profile_max = max(float(np.max(p_arr)) * 1.1, 1e-12)
        if profile_max > self.axes[0].get_ylim()[1] or profile_max < 0.35 * self.axes[0].get_ylim()[1]:
            self.axes[0].set_ylim(0.0, profile_max)
        self.accumulated_t.append(float(t_current))
        t_trace = np.array(self.accumulated_t)
        trace_max = 0.0
        for line, idx in self.trace_lines:
            self.accumulated_p[idx].append(float(p_arr[idx]))
            series = np.array(self.accumulated_p[idx])
            line.set_data(t_trace, series)
            if series.size:
                trace_max = max(trace_max, float(np.max(series)))
        trace_max = max(trace_max * 1.1, 1e-12)
        if trace_max > self.axes[1].get_ylim()[1] or trace_max < 0.35 * self.axes[1].get_ylim()[1]:
            self.axes[1].set_ylim(0.0, trace_max)
        tc = float(t_current)
        self.axes[0].set_title(f"Моделирование: p(r), t={tc:.2f} из {self.t_max:.2f} мкс")
        self.axes[1].set_title(f"Моделирование: p(t), t={tc:.2f} из {self.t_max:.2f} мкс")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(float(config.realtime_modeling_pause_sec))


def _wait_for_realtime_modeling_figures():
    backend_name = str(matplotlib.get_backend()).lower()
    if "agg" in backend_name:
        return
    open_figures = [fig for fig in LIVE_MODELING_FIGURES if plt.fignum_exists(fig.number)]
    if not open_figures:
        return
    print("Сохранение завершено. Закройте окно моделирования, чтобы завершить программу.")
    while any(plt.fignum_exists(fig.number) for fig in open_figures):
        plt.pause(0.1)
