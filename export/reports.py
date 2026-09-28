# -*- coding: utf-8 -*-
"""
export/reports.py - Текстовые отчёты: вывод на консоль и write_meeting_summary.
"""

import numpy as np
import config
from utils.formatting import _russian_value


# ---------------------------------------------------------------------------
# Отчёты на консоль
# ---------------------------------------------------------------------------

def report_pulse_summary(pulse_info):
    """Выводит информацию об импульсах: режим, частота, сегменты."""
    pulse_times = np.asarray(pulse_info.get("pulse_times", []), dtype=float)
    pulse_count = int(pulse_info.get("count", 0))
    segments = pulse_info.get("segments", [])
    pulse_amp = pulse_info.get("pulse_amount", None)
    print("Импульсы:")
    if pulse_amp is not None:
        print(f"  амплитуда импульса: {float(pulse_amp):.6g}")
    if "source_model" in pulse_info:
        print(f"  модель высвобождения: {_russian_value(pulse_info['source_model'])}")
    if "center_nm" in pulse_info and "sigma_nm" in pulse_info:
        print(
            f"  профиль подачи: центр={float(pulse_info['center_nm']):.1f} нм, "
            f"сигма={float(pulse_info['sigma_nm']):.1f} нм, окно={pulse_info.get('r_window')}"
        )
    if pulse_count <= 0 or pulse_times.size == 0:
        print("  расписание: нет импульсов")
        return
    t_start = float(pulse_times[0])
    t_end = float(pulse_times[-1])
    event_rate_hz = float(pulse_info.get("event_rate_hz", 0.0))
    repetition_hz = pulse_info.get("repetition_rate_hz", None)
    print(
        f"  начало={t_start:.2f} мкс, конец={t_end:.2f} мкс, "
        f"число={pulse_count}, средняя интенсивность в окне={event_rate_hz:.6g} Гц"
    )
    if repetition_hz is not None:
        print(
            f"  частота повторения по интервалам: {float(repetition_hz):.6g} Гц, "
            f"средний интервал={float(pulse_info.get('mean_gap_us', np.inf)):.2f} мкс"
        )
    else:
        print("  частота повторения по интервалам: один импульс, интервал не определён")
    time_window = pulse_info.get("time_window", None)
    if time_window is not None:
        print(f"  окно по времени: {float(min(time_window)):.2f}..{float(max(time_window)):.2f} мкс")
    max_rep_hz = float(
        pulse_info.get("max_repetition_frequency_hz",
                       pulse_info.get("max_frequency_hz", config.pulse_max_frequency_hz))
    )
    print(f"  физиологическое ограничение повторения: <= {max_rep_hz:.1f} Гц")
    if segments:
        print("  сегменты:")
        max_segments = 5
        clipped_any = any(bool(seg.get("clipped", False)) for seg in segments)
        if clipped_any:
            print(f"  ограничение частоты повторения: <= {max_rep_hz:.1f} Гц")
        for seg_idx, seg in enumerate(segments[:max_segments], start=1):
            seg_mode = _russian_value(seg.get("mode", "count"))
            seg_freq_hz = float(seg.get("frequency", 0.0)) * 1.0e6
            seg_rep_hz = float(seg.get("repetition_frequency", 0.0)) * 1.0e6
            requested_freq_hz = seg.get("requested_frequency_hz", None)
            requested_text = "" if requested_freq_hz is None else f", задано={float(requested_freq_hz):.6g} Гц"
            clipped_text = "да" if bool(seg.get("clipped", False)) else "нет"
            print(
                "   {idx}) {t_start:.2f}..{t_end:.2f} мкс, режим={mode}, число={count}, "
                "интенсивность={freq_hz:.6g} Гц, повторение={rep_hz:.6g} Гц{requested}, ограничено={clipped}".format(
                    idx=seg_idx,
                    t_start=float(seg.get("t_start", 0.0)),
                    t_end=float(seg.get("t_end", 0.0)),
                    mode=seg_mode,
                    count=int(seg.get("count", 0)),
                    freq_hz=seg_freq_hz,
                    rep_hz=seg_rep_hz,
                    requested=requested_text,
                    clipped=clipped_text,
                )
            )
        if len(segments) > max_segments:
            print(f"   ... еще сегментов: {len(segments) - max_segments}")


def report_normalization_summary(norm_ok, norm_min, norm_max, norm_dev):
    """Выводит результат проверки нормировки p(r,t)."""
    from solver.geometry import normalization_report_label  # ленивый импорт
    print("Нормировка:")
    print(f"  {'норма' if norm_ok else 'не норма'}")
    if norm_min is None or norm_max is None or norm_dev is None:
        print("  данных для проверки нет")
        return
    print(f"  {normalization_report_label()} по всем t: {norm_min:.6f}..{norm_max:.6f}")
    print(f"  максимум |сумма-1| = {norm_dev:.3e}")


def report_solver_convergence_summary(pulse_info):
    """Выводит сводку по сходимости решателя."""
    solver_info = pulse_info.get("solver_convergence", {})
    total_steps = int(solver_info.get("total_steps", 0))
    check_stride = int(solver_info.get("check_stride", 1))
    checked_steps = int(solver_info.get("checked_steps", total_steps))
    converged_steps = int(solver_info.get("converged_steps", 0))
    failed_steps = int(solver_info.get("failed_steps", 0))
    failed_times = np.asarray(solver_info.get("failed_times", []), dtype=float)
    residual_tol = float(solver_info.get("residual_tol", config.SOLVER_RESIDUAL_TOL))
    print("Сходимость решателя:")
    if check_stride > 1:
        print(
            f"  шагов всего: {total_steps} (невязка проверялась раз в {check_stride} "
            f"шагов -> {checked_steps} проверок), сошлось: {converged_steps}, "
            f"не сошлось: {failed_steps}, допуск={residual_tol:.1e}"
        )
    else:
        print(
            f"  шагов: {total_steps}, сошлось: {converged_steps}, "
            f"не сошлось: {failed_steps}, допуск={residual_tol:.1e}"
        )
    if failed_steps > 0 and failed_times.size:
        preview = ", ".join(f"{val:.2f}" for val in failed_times[:5])
        print(f"  первые t (мкс), где не сошлось: {preview}")


def report_parameter_summary():
    """Выводит ключевые параметры расчёта на консоль."""
    from solver.geometry import geometry_report_label  # ленивый импорт
    print("Параметры расчёта:")
    print(
        f"  D_cleft={config.D_cleft:.3g}, D_pm={config.D_pm:.3g} нм^2/мкс; "
        f"Omega_cleft={config.Omega_cleft:.3g}, Omega_pm={config.Omega_pm:.3g}, "
        f"Omega_pm_tripled={config.Omega_pm_tripled:.3g} нм/мкс"
    )
    print(
        f"  сетка: dr={config.dr:.3g} нм, dt={config.dt:.3g} мкс, t_max={config.t_max:.3g} мкс; "
        f"анимация={'вкл' if config.enable_animation else 'выкл'}, "
        f"экспорт={'вкл' if config.export_figures else 'выкл'}"
    )
    print(f"  геометрия: {geometry_report_label()}")
    print(
        "  интерпретация: "
        f"normalization_mode={config.normalization_mode} "
        f"(k_cleft={config.k_cleft:.3g}, k_pm={config.k_pm:.3g} 1/мкс)"
    )


def report_entropy_point_comparison(scenario, max_times=3):
    """Выводит локальную энтропию по контрольным точкам для первых нескольких моментов времени."""
    rows = scenario.get("entropy_point_comparison", [])
    if not rows:
        return
    times = []
    for row in rows:
        if row["time"] not in times:
            times.append(row["time"])
    print("Локальная энтропия по контрольным точкам:")
    for t_val in times[:max_times]:
        rows_t = [row for row in rows if row["time"] == t_val]
        if not rows_t:
            continue
        first = rows_t[0]
        print(
            f"  t={t_val:.2f} мкс: пик p при r={first['p_peak_radius']:.1f} нм, "
            f"пик -p ln p при r={first['density_entropy_peak_radius']:.1f} нм, "
            f"пик -q ln q при r={first['shell_entropy_peak_radius']:.1f} нм"
        )
        for row in sorted(rows_t, key=lambda item: item["radius"]):
            print(
                f"    r={row['radius']:.1f}: p={row['p']:.3e}, "
                f"-p ln p={row['local_entropy_density']:.3e}, "
                f"-q ln q={row['local_shell_entropy']:.3e}"
            )


def report_information_measure_comparison(scenarios):
    """Выводит KL-дивергенцию и взаимную информацию по парам точек для всех сценариев."""
    print("KL-дивергенция и взаимная информация по парам точек:")
    for scenario_name, scenario in scenarios.items():
        print(f"  {scenario_name}:")
        for item in scenario["info_measures"]:
            print(
                f"    {item['pair']}: симметричная KL-дивергенция={item['kl_sym']:.3e}, "
                f"взаимная информация={item['mutual_information']:.3e}, корреляция={item['correlation']:.3f}"
            )


def report_region_event_comparison(comparison):
    """Выводит редкие события по областям: среднее и частота превышений."""
    if not comparison:
        return
    print("Редкие события по областям:")
    for region_name, item in comparison.items():
        r_min, r_max = item["region"]
        print(
            f"  {region_name} ({r_min:.1f}..{r_max:.1f} нм): "
            f"среднее базового={item['reference_mean']:.3e}, среднее сравниваемого={item['candidate_mean']:.3e}, "
            f"превышения базового={item['reference_rate']:.3f}, превышения сравниваемого={item['candidate_rate']:.3f}"
        )


def report_scenario_region_kl(comparison):
    """Выводит KL-дивергенцию между сценариями по областям."""
    if not comparison:
        return
    print("KL между сценариями по областям:")
    for item in comparison:
        print(
            f"  {item['region']} ({item['r_min']:.1f}..{item['r_max']:.1f} нм): "
            f"симметричная KL-дивергенция={item['kl_sym']:.3e}, "
            f"среднее базового={item['reference_mean_mass']:.3e}, "
            f"среднее сравниваемого={item['candidate_mean_mass']:.3e}"
        )


def report_boundary_diagnostics(scenario):
    """Выводит диагностику правой границы сетки."""
    info = scenario.get("boundary_diagnostics", {})
    if not info:
        return
    ratio_center = info.get("max_edge_to_center_valid")
    ratio_center_text = "нет данных" if ratio_center is None else f"{ratio_center:.3e}"
    print("Правая граница:")
    print(
        f"  край r={info['edge_radius']:.1f} нм, центр r={info['center_radius']:.1f} нм; "
        f"максимум край/общий={info['max_edge_to_global']:.3e}, "
        f"максимум край/центр={ratio_center_text}"
    )


def report_pr_snapshots(r_values, t_values, p_history, times):
    """Выводит краткую сводку снимков p(r): пик и средний радиус."""
    from visualization.plots import _map_targets_to_indices  # ленивый импорт
    from solver.geometry import radial_cell_measures         # ленивый импорт

    idx_list, actual_times = _map_targets_to_indices(t_values, times)
    if idx_list.size == 0:
        return
    print("Снимки p(r):")
    for idx, t_actual in zip(idx_list, actual_times):
        profile = p_history[:, idx]
        peak_idx = int(np.argmax(profile))
        cell_m = radial_cell_measures(r_values, config.dr)
        total = float(np.sum(profile * cell_m))
        mean_r = 0.0 if total <= 0.0 else float(np.sum(profile * r_values * cell_m) / total)
        print(
            f" t={t_actual:.2f} мкс: максимум p={profile[peak_idx]:.3e} при r={r_values[peak_idx]:.1f} нм, "
            f"средний r={mean_r:.2f} нм"
        )


# ---------------------------------------------------------------------------
# Текстовый файл-сводка
# ---------------------------------------------------------------------------

def write_meeting_summary(
    output_root,
    scenarios,
    rare_event_comparison,
    theory_results,
    rare_region_comparison=None,
    scenario_region_kl=None,
):
    """
    Записывает текстовый файл сводка_расчёта.txt с ключевыми результатами
    по всем разделам: инверсия, энтропия, информационные меры, редкие события,
    импульсы, граница, теоретический скан.
    """
    from solver.geometry import shell_probability_report_formula  # ленивый импорт

    lines = []
    first_scenario = next(iter(scenarios.values())) if scenarios else None

    lines.append("Подготовка к следующей встрече")
    lines.append("")
    lines.append("1. Инверсия профиля")
    for scenario_name, scenario in scenarios.items():
        nonmono = scenario["nonmonotonic"]
        if nonmono is None:
            lines.append(f"- {scenario_name}: немонотонная инверсия (по максимуму p) не обнаружена")
        else:
            lines.append(
                f"- {scenario_name}: немонотонная инверсия (по максимуму p) при t={nonmono['time']:.2f} мкс, "
                f"пики r={', '.join(f'{val:.1f}' for val in nonmono['peak_radii'][:3])} нм"
            )
        mass_inv = scenario.get("mass_inversion")
        if mass_inv is None:
            lines.append(f"- {scenario_name}: инверсия по массе области (переходная зона/синапс) не обнаружена")
        else:
            mass_time, mass_ratio, mass_tr, mass_syn = mass_inv
            lines.append(
                f"- {scenario_name}: инверсия по массе области при t={mass_time:.2f} мкс, "
                f"отношение масс={mass_ratio:.3f} (масса перех. зоны={mass_tr:.3e}, масса синапса={mass_syn:.3e})"
            )
        exposure = scenario.get("excitotoxic_exposure")
        if exposure is None:
            lines.append(f"- {scenario_name}: внесинаптическая экспозиция не рассчитана (зона за радиусом пуста)")
        else:
            threshold_text = ""
            if "time_above_threshold" in exposure:
                threshold_text = (
                    f", время выше порога {exposure['concentration_threshold']:.3e} = "
                    f"{exposure['time_above_threshold']:.2f} мкс"
                )
            lines.append(
                f"- {scenario_name}: экспозиция за r={exposure['extrasynaptic_radius']:.1f} нм "
                f"(край синапса) - пик массы {exposure['peak_mass_extra']:.3e} при t={exposure['peak_time']:.2f} мкс, "
                f"накопленная доза x время за весь расчёт = {exposure['exposure_total']:.3e}{threshold_text}"
            )

    lines.append("")
    lines.append("2. Локальная энтропия")
    lines.append(
        f"- Используются две величины: плотность -p ln p и энтропия оболочек -q ln q, "
        f"где {shell_probability_report_formula()}."
    )
    lines.append("- При p < e^-1 функция -p ln p возрастает по p, поэтому максимум p не должен давать меньшую энтропию плотности.")
    lines.append("- Противоречие может возникать для энтропии оболочек: у больших r вклад усиливается геометрической мерой ячейки.")
    for scenario_name, scenario in scenarios.items():
        first_check = scenario["entropy_checks"][0] if scenario["entropy_checks"] else None
        if first_check is not None:
            lines.append(
                f"- {scenario_name}: при t={first_check['time']:.2f} мкс максимум p в r={first_check['peak_radius']:.1f} нм, "
                f"максимум энтропии оболочек в r={first_check['shell_entropy_peak_radius']:.1f} нм."
            )
        entropy_rows = scenario.get("entropy_point_comparison", [])
        if entropy_rows:
            target_row = min(entropy_rows, key=lambda row: abs(float(row["radius"]) - 265.0))
            lines.append(
                f"- {scenario_name}, контроль r={target_row['radius']:.1f} нм при t={target_row['time']:.2f} мкс: "
                f"p={target_row['p']:.3e}, -p ln p={target_row['local_entropy_density']:.3e}, "
                f"-q ln q={target_row['local_shell_entropy']:.3e}."
            )

    lines.append("")
    lines.append("3. Информационные меры")
    for scenario_name, scenario in scenarios.items():
        for item in scenario["info_measures"]:
            lines.append(
                f"- {scenario_name}, {item['pair']}: симметричная KL-дивергенция={item['kl_sym']:.3e}, "
                f"взаимная информация={item['mutual_information']:.3e}, корреляция={item['correlation']:.3f}"
            )
    if scenario_region_kl:
        lines.append("- KL между базовым и инвертированным сценариями по областям:")
        for item in scenario_region_kl:
            lines.append(
                f"  {item['region']} ({item['r_min']:.1f}..{item['r_max']:.1f} нм): "
                f"симметричная KL-дивергенция={item['kl_sym']:.3e}, "
                f"среднее базового={item['reference_mean_mass']:.3e}, "
                f"среднее сравниваемого={item['candidate_mean_mass']:.3e}."
            )

    lines.append("")
    lines.append("4. Редкие события на краях")
    if rare_event_comparison is not None:
        lines.append(
            f"- Порог массы хвоста взят по {config.rare_event_quantile:.0%}-квантилю базового сценария: "
            f"{rare_event_comparison['tail_threshold']:.3e}."
        )
        lines.append(
            f"- Частота превышений на хвосте: базовый={rare_event_comparison['reference_tail_rate']:.3f}, "
            f"сравниваемый={rare_event_comparison['candidate_tail_rate']:.3f}."
        )
        lines.append(
            f"- Частота превышений на краю: базовый={rare_event_comparison['reference_edge_rate']:.3f}, "
            f"сравниваемый={rare_event_comparison['candidate_edge_rate']:.3f}."
        )
    else:
        if first_scenario is not None:
            tail = first_scenario["rare_metrics"]
            lines.append(
                f"- Один сценарий: r >= {tail['tail_radius']:.1f} нм, "
                f"максимальная масса хвоста={np.max(tail['tail_mass_series']):.3e}."
            )
    if rare_region_comparison:
        for region_name, item in rare_region_comparison.items():
            r_min, r_max = item["region"]
            lines.append(
                f"- {region_name} ({r_min:.1f}..{r_max:.1f} нм): среднее базового={item['reference_mean']:.3e}, "
                f"среднее сравниваемого={item['candidate_mean']:.3e}, превышения базового={item['reference_rate']:.3f}, "
                f"превышения сравниваемого={item['candidate_rate']:.3f}."
            )

    lines.append("")
    lines.append("5. Импульсы")
    if first_scenario is not None:
        pulse_info = first_scenario["pulse_info"]
        lines.append(
            f"- Подача: модель={_russian_value(pulse_info.get('source_model', 'unknown'))}, "
            f"центр={float(pulse_info.get('center_nm', config.pulse_center_nm)):.1f} нм, "
            f"сигма={float(pulse_info.get('sigma_nm', config.pulse_sigma_nm)):.1f} нм, "
            f"окно={pulse_info.get('r_window')}."
        )
        lines.append(
            f"- Частота: число={pulse_info.get('count', 0)}, "
            f"интенсивность={float(pulse_info.get('event_rate_hz', 0.0)):.3g} Гц, "
            f"частота повторения={float(pulse_info.get('repetition_rate_hz', 0.0) or 0.0):.3g} Гц, "
            f"ограничение повторения <= {float(pulse_info.get('max_repetition_frequency_hz', config.pulse_max_frequency_hz)):.1f} Гц."
        )

    lines.append("")
    lines.append("6. Правая граница")
    for scenario_name, scenario in scenarios.items():
        edge = scenario.get("boundary_diagnostics", {})
        if edge:
            lines.append(
                f"- {scenario_name}: максимум край/общий={edge['max_edge_to_global']:.3e}, "
                f"край r={edge['edge_radius']:.1f} нм."
            )

    lines.append("")
    lines.append("7. Теоретический скан")
    if theory_results:
        for item in theory_results:
            if item["nonmonotonic"] is None:
                lines.append(f"- D_pm={item['D_pm']:.3g}, {item['case']}: инверсия не обнаружена")
            else:
                lines.append(
                    f"- D_pm={item['D_pm']:.3g}, {item['case']}: инверсия при t={item['nonmonotonic']['time']:.2f} мкс"
                )
    else:
        lines.append("- Отключён: используется только сценарий q = 3q слева.")

    with open(output_root / "сводка_расчёта.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
