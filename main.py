from pathlib import Path
import matplotlib.pyplot as plt

import config
from analysis.scenarios import (
    build_omega_case_table,
    run_scenario_analysis,
    select_comparison_scenarios,
    report_omega_case_behavior,
    report_theory_scan,
    run_theory_scan,
    compare_scenario_region_distributions,
)
from analysis.information import compare_rare_events, compare_region_events
from visualization.plots import (
    _sigma_label,
    _sigma_title,
    check_normalization,
    compute_probability_sums,
    plot_pr_snapshots,
    plot_sigma_lines,
)
from visualization.animations import (
    animate_pr_slices,
    animate_pt_series,
    animate_sigma_heatmap,
    animate_shannon_entropy,
    animate_local_entropy_at_radius,
)
from visualization.realtime import _wait_for_realtime_modeling_figures
from export.figures import export_scenario_figures, export_comparison_figures
from export.data import export_scenario_data
from export.reports import (
    report_parameter_summary,
    report_pulse_summary,
    report_normalization_summary,
    report_solver_convergence_summary,
    report_boundary_diagnostics,
    report_entropy_point_comparison,
    report_information_measure_comparison,
    report_pr_snapshots,
    report_region_event_comparison,
    report_scenario_region_kl,
    write_meeting_summary,
)


def main():
    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # --- Параметры ---
    report_parameter_summary()

    # --- Формируем таблицу сценариев по Omega ---
    omega_cases = build_omega_case_table(
        config.Omega_cleft,
        config.Omega_pm,
        include_comparison=config.compare_omega_cases,
    )

    # --- Запускаем все сценарии ---
    scenario_results = {}
    for case_name, case_cfg in omega_cases.items():
        scenario_results[case_name] = run_scenario_analysis(
            case_name,
            case_cfg["label"],
            case_cfg["omega_pm"],
            config.t_max,
        )

    # --- Консольный вывод по каждому сценарию ---
    for case_name, scenario in scenario_results.items():
        print("")
        print(f"=== Сценарий: {case_name} ({scenario['case_label']}) ===")
        report_pulse_summary(scenario["pulse_info"])

        if config.normalization_mode == "probability":
            norm_ok, norm_min, norm_max, norm_dev = check_normalization(
                scenario["p_history"],
                scenario["r_values"],
                config.dr,
            )
            report_normalization_summary(norm_ok, norm_min, norm_max, norm_dev)
        else:
            mass_series = compute_probability_sums(
                scenario["p_history"], scenario["r_values"], config.dr
            )
            print("Масса (режим concentration, ренормировка отключена):")
            print(
                f"  масса по всем t: {float(mass_series.min()):.6f}..{float(mass_series.max()):.6f} "
                "(отклонение от начальной массы ожидаемо - граница/отбор/импульсы)"
            )
        report_solver_convergence_summary(scenario["pulse_info"])

        if scenario.get("D_drop_start_radius") is not None:
            print(f"Начало спада D: r={scenario['D_drop_start_radius']:.1f} нм")

        report_boundary_diagnostics(scenario)

        if scenario["inv_profile"] is None:
            print("Переход переходная зона/синапс по отношению: не обнаружен")
        else:
            inv_time, inv_ratio, inv_tr_max, inv_syn_max = scenario["inv_profile"]
            print(
                "Переход переходная зона/синапс: t={:.2f} мкс, отношение={:.3f}, "
                "максимум переходной зоны={:.3e}, максимум синапса={:.3e}".format(
                    inv_time, inv_ratio, inv_tr_max, inv_syn_max
                )
            )

        if scenario["nonmonotonic"] is None:
            print("Немонотонная инверсия профиля: не обнаружена")
        else:
            print(
                f"Немонотонная инверсия профиля: t={scenario['nonmonotonic']['time']:.2f} мкс, "
                f"пики r={', '.join(f'{val:.1f}' for val in scenario['nonmonotonic']['peak_radii'][:3])} нм"
            )

        if scenario["peak_inversion"] is None:
            print("Инверсия пика: не обнаружена")
        else:
            print(
                f"Инверсия пика: t={scenario['peak_inversion'][0]:.2f} мкс, "
                f"радиус пика={scenario['peak_inversion'][1]:.1f} нм"
            )

        if scenario["mass_inversion"] is None:
            print("Инверсия по массе (переходная зона/синапс): не обнаружена")
        else:
            mass_time, mass_ratio, mass_tr, mass_syn = scenario["mass_inversion"]
            print(
                "Инверсия по массе (переходная зона/синапс): t={:.2f} мкс, отношение={:.3f}, "
                "масса переходной зоны={:.3e}, масса синапса={:.3e}".format(
                    mass_time, mass_ratio, mass_tr, mass_syn
                )
            )

        exposure = scenario.get("excitotoxic_exposure")
        if exposure is None:
            print("Внесинаптическая экспозиция: не рассчитана (зона за радиусом пуста)")
        else:
            print(
                "Внесинаптическая экспозиция (r>={:.1f} нм): пик массы={:.3e} при t={:.2f} мкс, "
                "накопленная доза x время={:.3e}".format(
                    exposure["extrasynaptic_radius"],
                    exposure["peak_mass_extra"],
                    exposure["peak_time"],
                    exposure["exposure_total"],
                )
            )

        if scenario["entropy_checks"]:
            first_check = scenario["entropy_checks"][0]
            print(
                f"Локальная энтропия при t={first_check['time']:.2f} мкс: "
                f"максимум p в r={first_check['peak_radius']:.1f} нм, "
                f"максимум энтропии оболочек в r={first_check['shell_entropy_peak_radius']:.1f} нм"
            )

        report_entropy_point_comparison(scenario)
        report_pr_snapshots(
            scenario["r_values"],
            scenario["t_values"],
            scenario["p_history"],
            scenario["pr_times_effective"],
        )

    # --- Сводный отчёт по сценариям ---
    comparison = [
        {
            "case": scenario_name,
            "label": scenario["case_label"],
            "omega_pm": scenario["omega_pm"],
            "nonmonotonic": scenario["nonmonotonic"],
        }
        for scenario_name, scenario in scenario_results.items()
    ]
    print("")
    if len(scenario_results) > 1:
        report_omega_case_behavior(comparison)
    else:
        only = comparison[0]
        print(
            f"Активный сценарий: {only['case']} ({only['label']}), "
            f"Omega_pm={only['omega_pm']:.3g}"
        )

    print("")
    report_information_measure_comparison(scenario_results)

    # --- Теоретический скан D_pm ---
    theory_results = []
    if config.theory_scan_enabled:
        theory_results = run_theory_scan(
            config.theory_scan_d_pm_values,
            omega_cases,
            config.max_r,
            config.dt,
            config.dr,
            config.theory_scan_t_max,
            config.D_cleft,
            config.xi_s,
            config.a,
            config.b,
            config.Omega_cleft,
            config.pulse_amount,
            config.pulse_r_window,
            config.num_pulses,
            config.pulse_segments,
            config.pulse_time_window,
            config.pulse_segment_default_mode,
            config.pulse_trigger_mode,
            config.pulse_trigger_radius,
            config.pulse_trigger_min_gap,
            config.pulse_trigger_slope_tol,
            config.pulse_trigger_require_rise,
            config.D_transition_kind,
            config.D_transition_steepness,
            config.Omega_transition_kind,
            config.Omega_transition_steepness,
            config.outer_boundary_mode,
            k_cleft=config.k_cleft,
            k_pm=config.k_pm,
            k_transition_kind=config.k_transition_kind,
            k_transition_steepness=config.k_transition_steepness,
            normalization_mode=config.normalization_mode,
        )
        print("")
        report_theory_scan(theory_results)

    # --- Сравнение сценариев: редкие события и KL ---
    reference_name, candidate_name = select_comparison_scenarios(scenario_results)
    rare_event_comparison = None
    rare_region_comparison = None
    scenario_region_kl = []

    if reference_name is not None and candidate_name is not None:
        reference_scenario = scenario_results[reference_name]
        candidate_scenario = scenario_results[candidate_name]

        rare_event_comparison = compare_rare_events(
            reference_scenario["rare_metrics"],
            candidate_scenario["rare_metrics"],
            quantile=config.rare_event_quantile,
        )
        rare_region_comparison = compare_region_events(
            reference_scenario["rare_region_metrics"],
            candidate_scenario["rare_region_metrics"],
            quantile=config.rare_event_quantile,
        )
        scenario_region_kl = compare_scenario_region_distributions(
            reference_scenario,
            candidate_scenario,
            config.scenario_kl_regions,
            time_window=config.information_time_window,
        )
        print("")
        print(
            "Редкие события на краях: "
            f"хвост базовый={rare_event_comparison['reference_tail_rate']:.3f}, "
            f"хвост сравниваемый={rare_event_comparison['candidate_tail_rate']:.3f}, "
            f"край базовый={rare_event_comparison['reference_edge_rate']:.3f}, "
            f"край сравниваемый={rare_event_comparison['candidate_edge_rate']:.3f}"
        )
        report_region_event_comparison(rare_region_comparison)
        print("")
        report_scenario_region_kl(scenario_region_kl)

    # --- Экспорт графиков и данных ---
    if config.export_figures or config.export_data:
        for scenario_name, scenario in scenario_results.items():
            scenario_dir = output_root / scenario_name
            if config.export_figures:
                export_scenario_figures(scenario, scenario_dir)
            export_scenario_data(scenario, scenario_dir)
        if len(scenario_results) > 1:
            export_comparison_figures(
                scenario_results,
                output_root / "сравнение",
                rare_event_comparison,
                scenario_region_kl,
            )
        write_meeting_summary(
            output_root,
            scenario_results,
            rare_event_comparison,
            theory_results,
            rare_region_comparison,
            scenario_region_kl,
        )
        print(f"Экспорт сохранён в {output_root.resolve()}")

    # --- Интерактивная анимация ---
    if config.enable_animation:
        active_case_name = (
            config.active_omega_case
            if config.active_omega_case in scenario_results
            else next(iter(scenario_results))
        )
        scenario = scenario_results[active_case_name]
        sigma_label = _sigma_label(config.sigma_kind)
        sigma_title = _sigma_title(config.sigma_kind)

        if config.pr_plot_mode == "overlay":
            plot_pr_snapshots(
                scenario["r_values"],
                scenario["t_values"],
                scenario["p_history"],
                scenario["pr_times_effective"],
                pause_sec=config.realtime_pause_sec,
            )
        else:
            animate_pr_slices(
                scenario["r_values"],
                scenario["t_values"],
                scenario["p_history"],
                scenario["pr_times_effective"],
                pause_sec=config.realtime_pause_sec,
                mode=config.pr_plot_mode,
                time_window=config.pr_realtime_window,
                stride=config.pr_realtime_stride,
                trail=config.pr_realtime_trail,
                trail_alpha=config.pr_realtime_trail_alpha,
                trail_max=config.pr_realtime_trail_max,
            )

        animate_pt_series(
            scenario["r_values"],
            scenario["t_values"],
            scenario["p_history"],
            scenario["pt_radii_effective"],
            pause_sec=config.realtime_pause_sec,
            pulse_times=scenario["pulse_info"].get("pulse_times", None),
            show_pulse_markers=config.show_pulse_markers,
            highlight_radii=(
                [scenario["D_drop_start_radius"]]
                if scenario.get("D_drop_start_radius") is not None
                else None
            ),
        )

        if config.sigma_plot_mode == "lines":
            plot_sigma_lines(
                scenario["r_values"],
                scenario["t_values"],
                scenario["sigma_history"],
                scenario["sigma_times_effective"],
                label=sigma_label,
                title=sigma_title,
                use_log=config.sigma_use_log,
                pause_sec=config.realtime_pause_sec,
            )
        elif config.show_sigma_heatmap:
            animate_sigma_heatmap(
                scenario["r_values"],
                scenario["t_values"],
                scenario["sigma_history"],
                config.sigma_r_window,
                config.sigma_t_window,
                pause_sec=config.realtime_pause_sec,
                label=sigma_label,
                title=sigma_title,
            )

        animate_shannon_entropy(
            scenario["t_values"],
            scenario["shannon_continuous"],
            pause_sec=config.realtime_pause_sec,
        )
        animate_local_entropy_at_radius(
            scenario["r_values"],
            scenario["t_values"],
            scenario["local_entropy_density"],
            config.local_entropy_radii,
            pause_sec=config.realtime_pause_sec,
        )

        plt.ioff()

    _wait_for_realtime_modeling_figures()


if __name__ == "__main__":
    main()
