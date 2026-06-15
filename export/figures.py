# -*- coding: utf-8 -*-
"""
export/figures.py - Экспорт matplotlib-фигур для одного сценария и для сравнения сценариев.
"""

import numpy as np
import matplotlib.pyplot as plt
import config
from utils.formatting import (
    _append_curve_rows,
    _append_event_rows,
    _append_bar_rows,
    _save_figure,
    _set_common_time_axis,
    _first_response_delay,
)
from utils.helpers import _append_time_unique
from visualization.plots import _map_targets_to_indices


def export_scenario_figures(scenario, scenario_dir):
    """
    Сохраняет полный набор графиков для одного сценария:
    профили p(r), локальная энтропия, p(t) по радиусам,
    энтропия оболочек, обзор импульсов, увеличенный отклик.
    """
    scenario_dir.mkdir(parents=True, exist_ok=True)
    r_values = scenario["r_values"]
    t_values = scenario["t_values"]
    p_history = scenario["p_history"]
    idx_list, actual_times = _map_targets_to_indices(t_values, scenario["pr_times_effective"])

    # --- Профили p(r) и энтропия плотности ---
    table_rows = []
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    for idx, t_actual in zip(idx_list, actual_times):
        label = f"t={t_actual:.0f} мкс"
        axes[0].plot(r_values, p_history[:, idx], lw=2, label=label)
        axes[1].plot(r_values, scenario["local_entropy_density"][:, idx], lw=2, label=label)
        _append_curve_rows(table_rows, "профиль p(r)", "r, нм", r_values, "p", p_history[:, idx], label)
        _append_curve_rows(
            table_rows, "энтропия плотности", "r, нм", r_values,
            "-p ln p", scenario["local_entropy_density"][:, idx], label,
        )
    axes[0].set_ylabel("p(r)")
    axes[0].set_title(f"Профили вероятности: {scenario['case_label']}")
    axes[0].grid(True)
    axes[0].legend()
    axes[1].set_xlabel("Радиус r, нм")
    axes[1].set_ylabel("-p ln p")
    axes[1].set_title("Локальная энтропия по плотности")
    axes[1].grid(True)
    axes[1].legend()
    _save_figure(fig, scenario_dir / "профили_вероятности_и_энтропия.png", table_rows)

    # --- p(t) по радиусам ---
    table_rows = []
    pulse_times = np.asarray(scenario["pulse_info"].get("pulse_times", []), dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    pulse_window = scenario["pulse_info"].get("time_window")
    if pulse_window is not None:
        pulse_start = max(float(t_values[0]), float(min(pulse_window)))
        pulse_end = min(float(t_values[-1]), float(max(pulse_window)))
        axes[0].axvspan(pulse_start, pulse_end, color="tab:red", alpha=0.08)
        axes[1].axvspan(pulse_start, pulse_end, color="tab:red", alpha=0.08)
    for pulse_time in pulse_times:
        axes[0].axvline(pulse_time, color="k", alpha=0.12, lw=0.8)
        axes[1].axvline(pulse_time, color="k", alpha=0.12, lw=0.8)
    for radius in scenario["pt_radii_effective"]:
        idx_r = int(np.argmin(np.abs(r_values - radius)))
        series = np.asarray(p_history[idx_r, :], dtype=float)
        delay = _first_response_delay(t_values, series, pulse_times)
        label = (
            f"r={r_values[idx_r]:.1f} нм" if delay is None
            else f"r={r_values[idx_r]:.1f} нм, задержка={delay:.0f} мкс"
        )
        axes[0].plot(t_values, series, lw=1.8, label=label)
        norm = series / max(float(np.max(series)), config.EPS)
        axes[1].plot(t_values, norm, lw=1.8, label=f"r={r_values[idx_r]:.1f} нм")
        _append_curve_rows(table_rows, "p(t) по радиусам", "t, мкс", t_values, "p", series, f"r={r_values[idx_r]:.1f} нм")
        _append_curve_rows(table_rows, "нормированная p(t) по радиусам", "t, мкс", t_values, "p / максимум p", norm, f"r={r_values[idx_r]:.1f} нм")
    _append_event_rows(table_rows, "p(t) по радиусам", pulse_times, series="импульс")
    axes[0].set_ylabel("p(t)")
    axes[0].set_title("p(t) при разных r")
    axes[0].grid(True)
    axes[0].legend()
    axes[1].set_xlabel("Время t, мкс")
    axes[1].set_ylabel("p(t) / максимум p(t)")
    axes[1].set_title("Нормированный отклик по радиусам")
    axes[1].grid(True)
    axes[1].legend()
    _set_common_time_axis(axes, t_values)
    _save_figure(fig, scenario_dir / "зависимость_вероятности_от_времени_по_радиусам.png", table_rows)

    # --- Энтропия оболочек и трассы ---
    table_rows = []
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    shell_idx_list, shell_times = _map_targets_to_indices(t_values, scenario["pr_times_effective"])
    for idx, t_actual in zip(shell_idx_list, shell_times):
        label = f"t={t_actual:.0f} мкс"
        axes[0].plot(r_values, scenario["local_shell_entropy"][:, idx], lw=2, label=label)
        _append_curve_rows(table_rows, "энтропия оболочек", "r, нм", r_values, "-q ln q", scenario["local_shell_entropy"][:, idx], label)
    for radius in scenario["pt_radii_effective"]:
        idx_r = int(np.argmin(np.abs(r_values - radius)))
        axes[1].plot(t_values, p_history[idx_r, :], lw=1.5, label=f"r={r_values[idx_r]:.1f} нм")
        _append_curve_rows(table_rows, "трассы в точках", "t, мкс", t_values, "p", p_history[idx_r, :], f"r={r_values[idx_r]:.1f} нм")
    for pulse_time in pulse_times:
        axes[1].axvline(pulse_time, color="k", alpha=0.12, lw=0.8)
    _append_event_rows(table_rows, "трассы в точках", pulse_times, series="импульс")
    axes[0].set_ylabel("-q ln q")
    axes[0].set_title("Локальная энтропия по вероятности оболочки")
    axes[0].grid(True)
    axes[0].legend()
    axes[1].set_xlabel("Время t, мкс")
    axes[1].set_ylabel("p(t)")
    axes[1].set_title("Трассы в контрольных точках")
    axes[1].grid(True)
    axes[1].legend()
    axes[1].set_xlim(float(t_values[0]), float(t_values[-1]))
    _save_figure(fig, scenario_dir / "энтропия_оболочек_и_трассы.png", table_rows)

    # --- Сравнение энтропии в контрольных точках ---
    entropy_rows = scenario.get("entropy_point_comparison", [])
    if entropy_rows:
        selected_times = []
        for candidate in [
            500.0 / config.time_divider,
            scenario["inv_profile"][0] if scenario["inv_profile"] is not None else None,
            5000.0 / config.time_divider,
        ]:
            if candidate is not None:
                selected_times = _append_time_unique(selected_times, float(candidate), tol=0.5 * config.dt)
        table_rows = []
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        for target_time in selected_times:
            rows_t = [
                row for row in entropy_rows
                if abs(float(row["time"]) - float(target_time)) <= 0.5 * config.dt + config.TIME_TOL
            ]
            if not rows_t:
                actual_time = min(entropy_rows, key=lambda row: abs(float(row["time"]) - float(target_time)))["time"]
                rows_t = [row for row in entropy_rows if row["time"] == actual_time]
            rows_t = sorted(rows_t, key=lambda row: row["radius"])
            radii_pts = [row["radius"] for row in rows_t]
            p_vals = [row["p"] for row in rows_t]
            density_vals = [row["local_entropy_density"] for row in rows_t]
            shell_vals = [row["local_shell_entropy"] for row in rows_t]
            label = f"t={rows_t[0]['time']:.0f} мкс"
            axes[0].plot(radii_pts, p_vals, marker="o", lw=2, label=label)
            axes[1].plot(radii_pts, density_vals, marker="o", lw=2, label=label)
            axes[2].plot(radii_pts, shell_vals, marker="o", lw=2, label=label)
            _append_curve_rows(table_rows, "p в точках", "r, нм", radii_pts, "p", p_vals, label)
            _append_curve_rows(table_rows, "энтропия плотности в точках", "r, нм", radii_pts, "-p ln p", density_vals, label)
            _append_curve_rows(table_rows, "энтропия оболочек в точках", "r, нм", radii_pts, "-q ln q", shell_vals, label)
        axes[0].set_ylabel("p")
        axes[0].set_title("Сравнение p и локальной энтропии по контрольным точкам")
        axes[1].set_ylabel("-p ln p")
        axes[2].set_ylabel("-q ln q")
        axes[2].set_xlabel("Радиус r, нм")
        for ax in axes:
            ax.grid(True)
            ax.legend()
        _save_figure(fig, scenario_dir / "сравнение_энтропии_в_точках.png", table_rows)

    # --- Обзор импульсов ---
    table_rows = []
    fig, axes = plt.subplots(4, 1, figsize=(10, 15))
    if pulse_times.size:
        axes[0].eventplot(pulse_times, colors="tab:red", lineoffsets=0.5, linelengths=0.8)
    _append_event_rows(table_rows, "времена импульсов", pulse_times, series="импульс")
    axes[0].set_xlim(t_values[0], t_values[-1])
    axes[0].set_yticks([])
    axes[0].set_title("Времена подачи импульсов")
    axes[0].grid(True, axis="x")
    axes[1].plot(r_values, scenario["pulse_shape"], lw=2.5, color="tab:orange")
    _append_curve_rows(table_rows, "форма импульса", "r, нм", r_values, "добавка к p", scenario["pulse_shape"], "один импульс")
    axes[1].set_title("Пространственный профиль одного импульса")
    axes[1].set_ylabel("добавка к p")
    axes[1].grid(True)
    propagation_times = []
    if pulse_times.size:
        first_pulse = float(pulse_times[0])
        for delta in (0.0, 50.0 / config.time_divider, 200.0 / config.time_divider, 1000.0 / config.time_divider):
            propagation_times.append(first_pulse + delta)
    else:
        propagation_times.extend([0.0, 50.0 / config.time_divider, 200.0 / config.time_divider, 1000.0 / config.time_divider])
    prop_idx, prop_times = _map_targets_to_indices(t_values, propagation_times)
    for idx, t_actual in zip(prop_idx, prop_times):
        axes[2].plot(r_values, p_history[:, idx], lw=2, label=f"t={t_actual:.0f} мкс")
        _append_curve_rows(table_rows, "профили после импульса", "r, нм", r_values, "p", p_history[:, idx], f"t={t_actual:.0f} мкс")
    axes[2].set_title("Распространение профиля после первого импульса")
    axes[2].set_ylabel("p(r)")
    axes[2].grid(True)
    axes[2].legend()
    axes[3].plot(t_values, scenario["rare_metrics"]["tail_mass_series"], lw=2, color="tab:green")
    _append_curve_rows(table_rows, "масса хвоста", "t, мкс", t_values, "масса хвоста", scenario["rare_metrics"]["tail_mass_series"], f"r>={scenario['rare_metrics']['tail_radius']:.0f} нм")
    axes[3].set_title(f"Масса в хвосте r >= {scenario['rare_metrics']['tail_radius']:.0f} нм")
    axes[3].set_xlabel("Время t, мкс")
    axes[3].set_ylabel("масса хвоста")
    axes[3].set_xlim(float(t_values[0]), float(t_values[-1]))
    axes[3].grid(True)
    _save_figure(fig, scenario_dir / "обзор_импульсов.png", table_rows)

    # --- Увеличенный отклик на первый импульс ---
    if pulse_times.size:
        zoom_start = float(t_values[0])
        zoom_end = float(t_values[-1])
        zoom_mask = (t_values >= zoom_start - config.TIME_TOL) & (t_values <= zoom_end + config.TIME_TOL)
        table_rows = []
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        visible_pulses = pulse_times[(pulse_times >= zoom_start - config.TIME_TOL) & (pulse_times <= zoom_end + config.TIME_TOL)]
        if visible_pulses.size:
            axes[0].eventplot(visible_pulses, colors="tab:red", lineoffsets=0.5, linelengths=0.8)
        _append_event_rows(table_rows, "окно импульсов", visible_pulses, series="импульс")
        axes[0].set_xlim(float(t_values[0]), float(t_values[-1]))
        axes[0].set_yticks([])
        axes[0].set_title("Окно подачи импульса")
        axes[0].grid(True, axis="x")
        trace_radii = [config.pulse_center_nm, 50.0, 200.0, 265.0]
        trace_idx, trace_actual = _map_targets_to_indices(r_values, trace_radii)
        for ri, r_actual in zip(trace_idx, trace_actual):
            axes[1].plot(t_values[zoom_mask], p_history[ri, zoom_mask], lw=2, label=f"r={r_actual:.1f} нм")
            _append_curve_rows(table_rows, "увеличенные трассы в точках", "t, мкс", t_values[zoom_mask], "p", p_history[ri, zoom_mask], f"r={r_actual:.1f} нм")
        for pulse_time in visible_pulses:
            axes[1].axvline(pulse_time, color="k", alpha=0.25, lw=0.9)
        axes[1].set_xlim(float(t_values[0]), float(t_values[-1]))
        axes[1].set_ylabel("p(t)")
        axes[1].set_title("Отклик в контрольных точках после импульса")
        axes[1].grid(True)
        axes[1].legend()
        first_pulse = float(pulse_times[0])
        response_times = [first_pulse + float(offset) for offset in config.pulse_response_offsets]
        response_idx, response_actual = _map_targets_to_indices(t_values, response_times)
        for idx, t_actual in zip(response_idx, response_actual):
            axes[2].plot(r_values, p_history[:, idx], lw=2, label=f"t={t_actual:.0f} мкс")
            _append_curve_rows(table_rows, "увеличенные профили отклика", "r, нм", r_values, "p", p_history[:, idx], f"t={t_actual:.0f} мкс")
        axes[2].set_xlabel("Радиус r, нм")
        axes[2].set_ylabel("p(r)")
        axes[2].set_title("Распространение профиля после подачи")
        axes[2].grid(True)
        axes[2].legend()
        _save_figure(fig, scenario_dir / "увеличенный_отклик_на_импульс.png", table_rows)


def export_comparison_figures(
    scenarios,
    comparison_dir,
    rare_event_comparison=None,
    scenario_region_kl=None,
):
    """
    Сохраняет графики, сравнивающие несколько сценариев:
    информационные меры, редкие события, KL по областям.
    """
    comparison_dir.mkdir(parents=True, exist_ok=True)
    scenario_names = list(scenarios.keys())
    pair_names = [item["pair"] for item in scenarios[scenario_names[0]]["info_measures"]]
    x = np.arange(len(pair_names))
    width = 0.35

    # --- KL и взаимная информация по парам точек ---
    table_rows = []
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    for offset, scenario_name in zip((-0.5, 0.5), scenario_names):
        measures = scenarios[scenario_name]["info_measures"]
        kl_vals = [item["kl_sym"] for item in measures]
        mi_vals = [item["mutual_information"] for item in measures]
        axes[0].bar(x + offset * width, kl_vals, width=width, label=scenario_name)
        axes[1].bar(x + offset * width, mi_vals, width=width, label=scenario_name)
        _append_bar_rows(table_rows, "KL по парам", pair_names, kl_vals, "симметричная KL-дивергенция", scenario_name)
        _append_bar_rows(table_rows, "взаимная информация по парам", pair_names, mi_vals, "взаимная информация", scenario_name)
    axes[0].set_ylabel("симметричная KL-дивергенция")
    axes[0].set_title("KL-дивергенция по парам точек")
    axes[0].grid(True, axis="y")
    axes[0].legend()
    axes[1].set_ylabel("взаимная информация")
    axes[1].set_title("Взаимная информация по парам точек")
    axes[1].grid(True, axis="y")
    axes[1].legend()
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(pair_names, rotation=15)
    _save_figure(fig, comparison_dir / "информационные_меры.png", table_rows)

    # --- Редкие события ---
    table_rows = []
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    for scenario_name, scenario in scenarios.items():
        axes[0].plot(scenario["t_values"], scenario["rare_metrics"]["tail_mass_series"], lw=2, label=scenario_name)
        axes[1].plot(scenario["t_values"], scenario["rare_metrics"]["edge_series"], lw=2, label=scenario_name)
        _append_curve_rows(table_rows, "масса хвоста", "t, мкс", scenario["t_values"], "масса хвоста", scenario["rare_metrics"]["tail_mass_series"], scenario_name)
        _append_curve_rows(table_rows, "вероятность крайней оболочки", "t, мкс", scenario["t_values"], "вероятность крайней оболочки", scenario["rare_metrics"]["edge_series"], scenario_name)
    if rare_event_comparison is not None:
        axes[0].axhline(rare_event_comparison["tail_threshold"], color="k", ls="--", alpha=0.5)
        axes[1].axhline(rare_event_comparison["edge_threshold"], color="k", ls="--", alpha=0.5)
        t0 = scenarios[scenario_names[0]]["t_values"]
        _append_curve_rows(table_rows, "масса хвоста", "t, мкс", [t0[0], t0[-1]], "порог хвоста", [rare_event_comparison["tail_threshold"]] * 2, "порог")
        _append_curve_rows(table_rows, "вероятность крайней оболочки", "t, мкс", [t0[0], t0[-1]], "порог края", [rare_event_comparison["edge_threshold"]] * 2, "порог")
    axes[0].set_ylabel("масса хвоста")
    axes[0].set_title("Редкие события: хвост распределения")
    axes[0].grid(True)
    axes[0].legend()
    axes[1].set_xlabel("Время t, мкс")
    axes[1].set_ylabel("вероятность крайней оболочки")
    axes[1].set_title("Редкие события: крайняя вероятность оболочки")
    axes[1].grid(True)
    axes[1].legend()
    _save_figure(fig, comparison_dir / "редкие_события.png", table_rows)

    # --- KL между сценариями по областям ---
    if scenario_region_kl:
        labels = [item["region"] for item in scenario_region_kl]
        kl_values = [item["kl_sym"] for item in scenario_region_kl]
        table_rows = []
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(np.arange(len(labels)), kl_values, color="tab:blue")
        _append_bar_rows(table_rows, "KL по областям", labels, kl_values, "симметричная KL-дивергенция", "сравнение сценариев")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=15)
        ax.set_ylabel("симметричная KL-дивергенция")
        ax.set_title("KL между сценариями по областям")
        ax.grid(True, axis="y")
        _save_figure(fig, comparison_dir / "дивергенция_по_областям_сценариев.png", table_rows)

    # --- Редкие события по областям ---
    region_names = list(next(iter(scenarios.values()))["rare_region_metrics"].keys())
    table_rows = []
    n_regions = len(region_names)
    fig, axes = plt.subplots(n_regions, 1, figsize=(10, 4 * n_regions), sharex=True)
    if n_regions == 1:
        axes = [axes]
    for ax, region_name in zip(axes, region_names):
        for scenario_name, scenario in scenarios.items():
            metrics = scenario["rare_region_metrics"][region_name]
            ax.plot(scenario["t_values"], metrics["mass_series"], lw=2, label=scenario_name)
            _append_curve_rows(table_rows, f"область {region_name}", "t, мкс", scenario["t_values"], "масса области", metrics["mass_series"], scenario_name)
        ax.set_ylabel("масса области")
        ax.set_title(f"Редкие события в области {region_name}")
        ax.grid(True)
        ax.legend()
    axes[-1].set_xlabel("Время t, мкс")
    _save_figure(fig, comparison_dir / "редкие_события_по_областям.png", table_rows)
