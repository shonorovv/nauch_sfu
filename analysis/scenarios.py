# -*- coding: utf-8 -*-
"""
analysis/scenarios.py - Управление сценариями моделирования.

Содержит:
- build_omega_case_table, select_comparison_scenarios
- boundary_diagnostics
- compare_scenario_region_distributions
- run_theory_scan, report_omega_case_behavior, report_theory_scan
- run_scenario_analysis - главный оркестратор
"""

import numpy as np
import config
from utils.helpers import _sanitize_window, _append_radius_unique, _append_time_unique
from analysis.inversion import (
    find_peak_inversion_time,
    find_profile_inversion_time,
    find_nonmonotonic_inversion_time,
)
from analysis.entropy import (
    compute_local_entropy_fields,
    compute_sigma_history,
    analyze_local_entropy_consistency,
    compare_entropy_at_points,
)
from analysis.information import (
    kl_divergence,
    compute_information_measures,
    compute_rare_event_metrics,
    compute_region_event_metrics,
)


# ---------------------------------------------------------------------------
# Таблица случаев Омега и выбор пар для сравнения
# ---------------------------------------------------------------------------

def build_omega_case_table(omega_cleft, omega_pm_baseline, include_comparison=True):
    omega_cleft = float(omega_cleft)
    return {
        "q_3q_слева": {
            "label": "q = 3q слева",
            "omega_pm": 3.0 * omega_cleft,
        }
    }


def select_comparison_scenarios(scenarios, reference_name="базовый", candidate_name="утроенный"):
    if len(scenarios) < 2:
        return None, None
    if reference_name not in scenarios:
        reference_name = next(iter(scenarios))
    if candidate_name not in scenarios or candidate_name == reference_name:
        candidate_name = next(name for name in scenarios if name != reference_name)
    return reference_name, candidate_name


# ---------------------------------------------------------------------------
# Диагностика граничных условий
# ---------------------------------------------------------------------------

def boundary_diagnostics(p_history, r_values, center_radius=0.0):
    """Проверяет, не загрязняет ли граница распределение p(r,t)."""
    if p_history.size == 0:
        return {}
    r_values = np.asarray(r_values, dtype=float)
    center_idx = int(np.argmin(np.abs(r_values - float(center_radius))))
    edge_idx = int(r_values.size - 1)
    center_series = np.asarray(p_history[center_idx, :], dtype=float)
    edge_series = np.asarray(p_history[edge_idx, :], dtype=float)
    global_series = np.max(np.asarray(p_history, dtype=float), axis=0)
    center_floor = max(float(np.max(center_series)) * 1e-8, config.EPS)
    valid_center = center_series >= center_floor
    edge_to_center = edge_series[valid_center] / np.maximum(center_series[valid_center], config.EPS)
    edge_to_global = edge_series / np.maximum(global_series, config.EPS)
    return {
        "center_radius": float(r_values[center_idx]),
        "edge_radius": float(r_values[edge_idx]),
        "edge_density_max": float(np.max(edge_series)),
        "center_density_max": float(np.max(center_series)),
        "global_density_max": float(np.max(global_series)),
        "max_edge_to_global": float(np.max(edge_to_global)),
        "max_edge_to_center_valid": float(np.max(edge_to_center)) if edge_to_center.size else None,
    }


# ---------------------------------------------------------------------------
# Сравнение сценариев по KL-дивергенции в областях
# ---------------------------------------------------------------------------

def compare_scenario_region_distributions(reference, candidate, regions, time_window=None):
    """Побластное сравнение двух сценариев через KL-дивергенцию."""
    results = []
    ref_t = np.asarray(reference["t_values"], dtype=float)
    cand_t = np.asarray(candidate["t_values"], dtype=float)
    if time_window is None:
        ref_mask = np.ones(ref_t.shape, dtype=bool)
        cand_mask = np.ones(cand_t.shape, dtype=bool)
    else:
        ref_start, ref_end = _sanitize_window(time_window, float(ref_t[0]), float(ref_t[-1]))
        cand_start, cand_end = _sanitize_window(time_window, float(cand_t[0]), float(cand_t[-1]))
        ref_mask = (ref_t >= ref_start - config.TIME_TOL) & (ref_t <= ref_end + config.TIME_TOL)
        cand_mask = (cand_t >= cand_start - config.TIME_TOL) & (cand_t <= cand_end + config.TIME_TOL)
    for region_name, region in regions.items():
        ref_series = compute_region_event_metrics(
            reference["shell_probabilities"], reference["r_values"], region
        )["mass_series"]
        cand_series = compute_region_event_metrics(
            candidate["shell_probabilities"], candidate["r_values"], region
        )["mass_series"]
        ref_values = np.asarray(ref_series[ref_mask], dtype=float)
        cand_values = np.asarray(cand_series[cand_mask], dtype=float)
        if ref_values.size != cand_values.size and ref_values.size > 1 and cand_values.size > 1:
            cand_values = np.interp(ref_t[ref_mask], cand_t[cand_mask], cand_values)
        count = min(ref_values.size, cand_values.size)
        ref_values = ref_values[:count]
        cand_values = cand_values[:count]
        kl_ref_candidate = kl_divergence(ref_values, cand_values)
        kl_candidate_ref = kl_divergence(cand_values, ref_values)
        results.append({
            "region": region_name,
            "r_min": float(min(region)),
            "r_max": float(max(region)),
            "reference_case": reference["case_name"],
            "candidate_case": candidate["case_name"],
            "kl_reference_candidate": kl_ref_candidate,
            "kl_candidate_reference": kl_candidate_ref,
            "kl_sym": 0.5 * (kl_ref_candidate + kl_candidate_ref),
            "reference_mean_mass": float(np.mean(ref_values)) if count else 0.0,
            "candidate_mean_mass": float(np.mean(cand_values)) if count else 0.0,
        })
    return results


# ---------------------------------------------------------------------------
# Отчёты о сканировании параметров (print)
# ---------------------------------------------------------------------------

def report_omega_case_behavior(comparison):
    print("Сравнение сценариев Омега:")
    for item in comparison:
        nonmono = item["nonmonotonic"]
        if nonmono is None:
            print(
                f"  {item['case']}: {item['label']}, Omega_pm={item['omega_pm']:.3g} -> "
                "немонотонная инверсия не обнаружена"
            )
            continue
        peaks = ", ".join(f"{val:.1f}" for val in nonmono["peak_radii"][:3])
        troughs = ", ".join(f"{val:.1f}" for val in nonmono["trough_radii"][:3])
        print(
            f"  {item['case']}: {item['label']}, Omega_pm={item['omega_pm']:.3g} -> "
            f"инверсия при t={nonmono['time']:.2f} мкс; пики r={peaks} нм; впадины r={troughs} нм"
        )


def report_theory_scan(results):
    print("Скан по параметрам D_pm и Omega_pm:")
    current_d = None
    for item in results:
        if current_d != item["D_pm"]:
            current_d = item["D_pm"]
            print(f"  D_pm = {current_d:.3g} нм^2/мкс")
        nonmono = item["nonmonotonic"]
        if nonmono is None:
            print(f"    {item['case']}: инверсия не обнаружена")
        else:
            print(
                f"    {item['case']}: инверсия при t={nonmono['time']:.2f} мкс "
                f"(Omega_pm={item['omega_pm']:.3g})"
            )


# ---------------------------------------------------------------------------
# Теоретическое сканирование по D_pm и Omega
# ---------------------------------------------------------------------------

def run_theory_scan(
    d_pm_values,
    omega_cases,
    max_r,
    dt,
    dr,
    t_max,
    D_cleft,
    xi_s,
    a,
    b,
    Omega_cleft,
    pulse_amount,
    pulse_r_window,
    num_pulses,
    pulse_segments,
    pulse_time_window,
    pulse_segment_default_mode,
    pulse_trigger_mode,
    pulse_trigger_radius,
    pulse_trigger_min_gap,
    pulse_trigger_slope_tol,
    pulse_trigger_require_rise,
    D_transition_kind,
    D_transition_steepness,
    Omega_transition_kind,
    Omega_transition_steepness,
    outer_boundary_mode,
):
    from solver.fp_solver import solve_fp_equation
    results = []
    for d_pm_case in d_pm_values:
        for case_name, case_cfg in omega_cases.items():
            p_history_case, r_case, t_case, _, _, _ = solve_fp_equation(
                max_r, dt, dr, t_max, D_cleft, float(d_pm_case), xi_s, a, b,
                float(Omega_cleft), float(case_cfg["omega_pm"]),
                pulse_amount, pulse_r_window, num_pulses, pulse_segments, pulse_time_window,
                pulse_segment_default_mode, pulse_trigger_mode, pulse_trigger_radius,
                pulse_trigger_min_gap, pulse_trigger_slope_tol, pulse_trigger_require_rise,
                D_transition_kind, D_transition_steepness,
                Omega_transition_kind, Omega_transition_steepness,
                outer_boundary_mode=outer_boundary_mode,
            )
            nonmono = find_nonmonotonic_inversion_time(
                r_case, t_case, p_history_case,
                peak_floor_fraction=config.inversion_peak_floor_fraction,
                boundary_guard_nm=config.inversion_boundary_guard_nm,
            )
            results.append({
                "case": case_name,
                "label": case_cfg["label"],
                "D_pm": float(d_pm_case),
                "omega_pm": float(case_cfg["omega_pm"]),
                "nonmonotonic": nonmono,
            })
    return results


# ---------------------------------------------------------------------------
# Главная функция сценария
# ---------------------------------------------------------------------------

def run_scenario_analysis(case_name, case_label, omega_pm_value, sim_t_max):
    """
    Запускает Fokker-Planck для заданного сценария (omega_pm_value),
    вычисляет всю аналитику и возвращает словарь результатов.

    Параметры берутся из config. Для временного переопределения параметров
    используй паттерн: setattr(config, 'param_name', new_value) до вызова.
    """
    from solver.fp_solver import solve_fp_equation
    from physics.transport import find_transition_start_radius
    from physics.pulses import pulse_increment_profile
    from solver.geometry import normalize_geometry_mode

    p_history, r_values, t_values, D_values, Omega_values, pulse_info = solve_fp_equation(
        config.max_r,
        config.dt,
        config.dr,
        sim_t_max,
        config.D_cleft,
        config.D_pm,
        config.xi_s,
        config.a,
        config.b,
        float(config.Omega_cleft),
        float(omega_pm_value),
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
        outer_boundary_mode=config.outer_boundary_mode,
    )

    entropy_bundle = compute_local_entropy_fields(p_history, r_values, config.dr)
    sigma_history = compute_sigma_history(
        p_history,
        D_values,
        Omega_values,
        config.dr,
        kind=config.sigma_kind,
        local_entropy=entropy_bundle["local_entropy_density"],
        p_min=config.sigma_p_min,
        d_min=config.sigma_d_min,
        r_values=r_values,
    )

    pr_times_effective = list(config.pr_times)
    sigma_times_effective = list(config.sigma_times)
    pt_radii_effective = list(config.pt_radii)
    entropy_probe_radii_effective = list(config.entropy_probe_radii)

    d_drop_start_radius = None
    if config.auto_include_drop_radius:
        d_drop_start_radius = find_transition_start_radius(
            r_values,
            D_values,
            config.D_cleft,
            config.D_pm,
            fraction=config.D_drop_start_fraction,
        )
        if d_drop_start_radius is not None:
            pt_radii_effective = _append_radius_unique(
                pt_radii_effective, d_drop_start_radius, tol=0.5 * config.dr
            )
            entropy_probe_radii_effective = _append_radius_unique(
                entropy_probe_radii_effective, d_drop_start_radius, tol=0.5 * config.dr
            )

    inv_start, inv_end = config.inversion_zone
    inv_profile = find_profile_inversion_time(
        r_values, t_values, p_history,
        inv_start, inv_end,
        min_ratio=config.inversion_min_ratio,
        pick="first",
        peak_floor_fraction=config.inversion_peak_floor_fraction,
        boundary_guard_nm=config.inversion_boundary_guard_nm,
    )
    nonmonotonic = find_nonmonotonic_inversion_time(
        r_values, t_values, p_history,
        peak_floor_fraction=config.inversion_peak_floor_fraction,
        boundary_guard_nm=config.inversion_boundary_guard_nm,
    )
    peak_inversion = find_peak_inversion_time(
        r_values, t_values, p_history,
        config.inversion_reference_radius,
        boundary_guard_nm=config.inversion_boundary_guard_nm,
    )

    if config.auto_include_inversion_time and inv_profile is not None:
        pr_times_effective = _append_time_unique(pr_times_effective, inv_profile[0], tol=0.5 * config.dt)
        sigma_times_effective = _append_time_unique(sigma_times_effective, inv_profile[0], tol=0.5 * config.dt)
    if config.auto_include_inversion_time and nonmonotonic is not None:
        pr_times_effective = _append_time_unique(pr_times_effective, nonmonotonic["time"], tol=0.5 * config.dt)
        sigma_times_effective = _append_time_unique(sigma_times_effective, nonmonotonic["time"], tol=0.5 * config.dt)

    target_entropy_times = list(pr_times_effective)
    entropy_checks = analyze_local_entropy_consistency(
        r_values, t_values, p_history,
        entropy_bundle["local_entropy_density"],
        entropy_bundle["local_shell_entropy"],
        target_entropy_times,
    )

    entropy_point_times = list(config.entropy_probe_times)
    for t_val in pr_times_effective:
        entropy_point_times = _append_time_unique(entropy_point_times, float(t_val), tol=0.5 * config.dt)
    entropy_point_comparison = compare_entropy_at_points(
        r_values, t_values, p_history,
        entropy_bundle["local_entropy_density"],
        entropy_bundle["local_shell_entropy"],
        entropy_bundle["shell_probabilities"],
        entropy_probe_radii_effective,
        entropy_point_times,
    )

    info_measures = compute_information_measures(
        t_values, r_values, entropy_bundle["shell_probabilities"],
        config.information_pairs,
        time_window=config.information_time_window,
        bins=config.information_hist_bins,
    )

    rare_metrics = compute_rare_event_metrics(
        entropy_bundle["shell_probabilities"],
        r_values,
        config.rare_event_radius,
    )
    rare_region_metrics = {
        "synapse_edge": compute_region_event_metrics(
            entropy_bundle["shell_probabilities"], r_values, config.rare_synapse_edge_band
        ),
        "outer_tail": compute_region_event_metrics(
            entropy_bundle["shell_probabilities"], r_values, (config.rare_event_radius, config.max_r)
        ),
    }

    return {
        "case_name": case_name,
        "case_label": case_label,
        "omega_pm": float(omega_pm_value),
        "geometry_mode": pulse_info.get("geometry_mode", normalize_geometry_mode()),
        "p_history": p_history,
        "r_values": r_values,
        "t_values": t_values,
        "D_values": D_values,
        "Omega_values": Omega_values,
        "pulse_info": pulse_info,
        "sigma_history": sigma_history,
        "pr_times_effective": sorted(pr_times_effective),
        "sigma_times_effective": sorted(sigma_times_effective),
        "pt_radii_effective": pt_radii_effective,
        "entropy_probe_radii_effective": entropy_probe_radii_effective,
        "D_drop_start_radius": d_drop_start_radius,
        "local_entropy_density": entropy_bundle["local_entropy_density"],
        "local_shell_entropy": entropy_bundle["local_shell_entropy"],
        "shell_probabilities": entropy_bundle["shell_probabilities"],
        "shannon_continuous": entropy_bundle["shannon_continuous"],
        "shannon_shell": entropy_bundle["shannon_shell"],
        "entropy_checks": entropy_checks,
        "entropy_point_comparison": entropy_point_comparison,
        "info_measures": info_measures,
        "rare_metrics": rare_metrics,
        "rare_region_metrics": rare_region_metrics,
        "boundary_diagnostics": boundary_diagnostics(p_history, r_values, config.pulse_center_nm),
        "inv_profile": inv_profile,
        "nonmonotonic": nonmonotonic,
        "peak_inversion": peak_inversion,
        "pulse_shape": pulse_increment_profile(
            r_values, config.pulse_r_window, config.pulse_amount, config.dr,
            pulse_info.get("geometry_mode")
        ),
    }
