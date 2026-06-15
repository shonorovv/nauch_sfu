# -*- coding: utf-8 -*-
import numpy as np
import backend

import config
from solver.geometry import (
    normalize_geometry_mode,
    radial_cell_measures,
    radial_face_measures,
)
from physics.transport import D_variable, Omega_variable
from physics.pulses import apply_pulse, compute_pulse_schedule


def solve_fp_equation(
    max_r,
    dt,
    dr,
    t_max,
    D_cleft,
    D_pm,
    xi_s,
    a,
    b,
    Omega_cleft,
    Omega_pm,
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
    outer_boundary_mode="dirichlet",
):
    if dt <= 0.0 or dr <= 0.0:
        raise ValueError("dt и dr должны быть положительными.")
    boundary_mode = str(outer_boundary_mode).lower()
    if boundary_mode not in ("dirichlet", "no_flux"):
        raise ValueError(f"Неизвестное условие на внешней границе: {outer_boundary_mode}")

    n_r = max(int(np.round(max_r / dr)), 1)
    r_values = (np.arange(n_r) + 0.5) * dr
    r_faces = np.arange(n_r + 1) * dr
    t_values = np.arange(0.0, t_max + 0.5 * dt, dt)
    n_t = len(t_values)
    geometry_kind = normalize_geometry_mode(config.geometry_mode)
    dV = radial_cell_measures(r_values, dr, geometry_kind)

    D_values = D_variable(r_values, D_cleft, D_pm, xi_s, a, b, D_transition_kind, D_transition_steepness)
    D_faces = D_variable(r_faces, D_cleft, D_pm, xi_s, a, b, D_transition_kind, D_transition_steepness)
    Omega_values = Omega_variable(r_values, Omega_cleft, Omega_pm, xi_s, a, b, Omega_transition_kind, Omega_transition_steepness)
    Omega_faces = Omega_variable(r_faces, Omega_cleft, Omega_pm, xi_s, a, b, Omega_transition_kind, Omega_transition_steepness)
    Omega_faces[0] = 0.0
    if boundary_mode == "no_flux":
        Omega_faces[-1] = 0.0

    face_measures = radial_face_measures(r_faces, geometry_kind)
    diff_faces = face_measures * D_faces / dr
    adv_faces = face_measures * Omega_faces
    inv_dt = dV / dt

    int_df = diff_faces[1:n_r]
    int_af = adv_faces[1:n_r]
    lower_diag = np.where(int_af >= 0.0, -(int_df + int_af), -int_df)
    upper_diag = np.where(int_af >= 0.0, -int_df, -int_df + int_af)
    main_from_left = np.where(int_af >= 0.0, int_df, int_df - int_af)
    main_from_right = np.where(int_af >= 0.0, int_df + int_af, int_df)
    main_diag = inv_dt.copy()
    main_diag[1:] += main_from_left
    main_diag[:-1] += main_from_right
    if boundary_mode == "dirichlet":
        diff_bc = 2.0 * face_measures[-1] * D_faces[-1] / dr
        adv_bc = max(float(face_measures[-1] * Omega_faces[-1]), 0.0)
        main_diag[-1] += diff_bc + adv_bc

    matrix = backend.diags(
        [backend.to_gpu(lower_diag), backend.to_gpu(main_diag), backend.to_gpu(upper_diag)],
        offsets=[-1, 0, 1],
        format="csc",
    )
    solver_fn = backend.factorized(matrix)

    # --- Настройка импульсов ---
    trigger_mode = str(pulse_trigger_mode).lower()
    adaptive_on_decay = trigger_mode in ("on_decay", "decay", "adaptive")
    pulse_segments_info = []
    pulse_indices_used = []
    pulse_mask = np.zeros(n_t, dtype=bool)

    adaptive_probe_idx = None
    adaptive_probe_radius = None
    adaptive_probe_prev = 0.0
    adaptive_probe_curr = 0.0
    adaptive_saw_rise = False
    adaptive_max_pulses = max(int(num_pulses), 0)
    adaptive_pulses_fired = 0
    adaptive_last_pulse_idx = -(10 ** 9)
    adaptive_window_start_idx = 0
    adaptive_window_end_idx = n_t - 1
    adaptive_slope_tol = max(float(pulse_trigger_slope_tol), 0.0)
    adaptive_require_rise = bool(pulse_trigger_require_rise)

    max_freq_per_us = max(float(config.pulse_max_frequency_hz), 0.0) / 1.0e6
    phys_min_gap_us = 0.0 if max_freq_per_us <= 0.0 else 1.0 / max_freq_per_us
    effective_min_gap_us = max(float(pulse_trigger_min_gap), phys_min_gap_us)
    adaptive_min_gap_steps = max(
        int(np.ceil(max(effective_min_gap_us, 0.0) / max(float(dt), config.TIME_TOL))),
        0,
    )

    if adaptive_on_decay:
        adaptive_probe_idx = int(np.argmin(np.abs(r_values - float(pulse_trigger_radius))))
        adaptive_probe_radius = float(r_values[adaptive_probe_idx])
        if pulse_time_window is not None:
            t_start = max(t_values[0], float(min(pulse_time_window)))
            t_end = min(t_values[-1], float(max(pulse_time_window)))
            if t_end >= t_start:
                adaptive_window_start_idx = int(np.searchsorted(t_values, t_start, side="left"))
                adaptive_window_end_idx = int(np.searchsorted(t_values, t_end, side="right")) - 1
            else:
                adaptive_window_start_idx = 1
                adaptive_window_end_idx = 0
        adaptive_window_start_idx = int(np.clip(adaptive_window_start_idx, 0, n_t - 1))
        adaptive_window_end_idx = int(np.clip(adaptive_window_end_idx, -1, n_t - 1))
        if (
            adaptive_max_pulses > 0
            and adaptive_window_start_idx <= adaptive_window_end_idx
            and adaptive_window_start_idx < n_t
        ):
            pulse_mask[adaptive_window_start_idx] = True
    else:
        pulse_indices, pulse_segments_info = compute_pulse_schedule(
            t_values,
            num_pulses,
            pulse_segments,
            pulse_time_window,
            segment_default_mode=pulse_segment_default_mode,
        )
        pulse_mask[pulse_indices] = True

    p_previous = np.zeros_like(r_values)
    if pulse_mask[0]:
        p_previous = apply_pulse(
            p_previous, r_values, pulse_r_window, pulse_amount, dr, geometry_kind
        )
        pulse_indices_used.append(0)
        if adaptive_on_decay:
            adaptive_pulses_fired = 1
            adaptive_last_pulse_idx = 0

    # Хранение истории с прореживанием: не более max_saved_frames кадров
    max_saved = max(int(getattr(config, 'max_saved_frames', 2000)), 1)
    save_stride = max(1, n_t // max_saved)
    n_alloc = max_saved + 2
    p_history = np.zeros((n_r, n_alloc), dtype=np.float64)
    t_saved = np.zeros(n_alloc, dtype=np.float64)
    p_history[:, 0] = p_previous
    t_saved[0] = t_values[0]
    save_count = 1

    from visualization.realtime import RealtimeModelingPlot
    live_plot = RealtimeModelingPlot(
        r_values,
        t_values,
        p_previous,
        pulse_mask,
        pulse_time_window,
        config.pt_radii,
    )

    # Перенос горячих массивов на устройство (GPU или CPU)
    dV_dev = backend.to_gpu(dV)
    inv_dt_dev = backend.to_gpu(inv_dt)
    p_previous_dev = backend.to_gpu(p_previous)
    rhs_dev = backend.xp.empty(n_r)

    if adaptive_on_decay and adaptive_probe_idx is not None:
        adaptive_probe_prev = float(p_previous_dev[adaptive_probe_idx])
        adaptive_probe_curr = adaptive_probe_prev

    solver_total_steps = max(n_t - 1, 0)
    solver_converged_steps = 0
    solver_failed_times = []

    for t_idx in range(1, n_t):
        if (
            adaptive_on_decay
            and adaptive_probe_idx is not None
            and not pulse_mask[t_idx]
            and adaptive_pulses_fired < adaptive_max_pulses
            and adaptive_window_start_idx <= t_idx <= adaptive_window_end_idx
        ):
            slope = adaptive_probe_curr - adaptive_probe_prev
            if slope > adaptive_slope_tol:
                adaptive_saw_rise = True
            can_fire = (t_idx - adaptive_last_pulse_idx) >= adaptive_min_gap_steps
            if can_fire and slope < -adaptive_slope_tol and (
                adaptive_saw_rise or not adaptive_require_rise
            ):
                pulse_mask[t_idx] = True

        if pulse_mask[t_idx]:
            p_cpu = backend.to_cpu(p_previous_dev)
            p_cpu = apply_pulse(p_cpu, r_values, pulse_r_window, pulse_amount, dr, geometry_kind)
            p_previous_dev = backend.to_gpu(p_cpu)
            pulse_indices_used.append(t_idx)
            if adaptive_on_decay:
                adaptive_pulses_fired += 1
                adaptive_last_pulse_idx = t_idx
                adaptive_saw_rise = False
                adaptive_probe_prev = float(p_previous_dev[adaptive_probe_idx])
                adaptive_probe_curr = adaptive_probe_prev

        backend.xp.multiply(inv_dt_dev, p_previous_dev, out=rhs_dev)
        p_current_dev = solver_fn(rhs_dev)

        residual_ok = False
        if bool(backend.xp.all(backend.xp.isfinite(p_current_dev))):
            residual_dev = matrix.dot(p_current_dev) - rhs_dev
            denom = max(float(backend.xp.linalg.norm(rhs_dev)), 1e-30)
            residual_rel = float(backend.xp.linalg.norm(residual_dev) / denom)
            residual_ok = residual_rel <= float(config.SOLVER_RESIDUAL_TOL)
        if residual_ok:
            solver_converged_steps += 1
        else:
            solver_failed_times.append(float(t_values[t_idx]))

        backend.xp.maximum(p_current_dev, 0.0, out=p_current_dev)
        total = float(backend.xp.dot(p_current_dev, dV_dev))
        if total > 0.0:
            p_current_dev /= total

        if t_idx % save_stride == 0 or t_idx == n_t - 1:
            if save_count < n_alloc:
                p_history[:, save_count] = backend.to_cpu(p_current_dev)
                t_saved[save_count] = t_values[t_idx]
                save_count += 1

        if adaptive_on_decay and adaptive_probe_idx is not None:
            adaptive_probe_prev = adaptive_probe_curr
            adaptive_probe_curr = float(p_current_dev[adaptive_probe_idx])

        if live_plot.enabled and (
            t_idx == n_t - 1
            or t_idx % max(int(config.realtime_modeling_stride), 1) == 0
            or pulse_mask[t_idx]
        ):
            live_plot.update(float(t_values[t_idx]), backend.to_cpu(p_current_dev))

        p_previous_dev = p_current_dev

    p_history = p_history[:, :save_count]
    t_saved = t_saved[:save_count]

    # --- Сборка pulse_info ---
    pulse_indices_arr = np.array(pulse_indices_used, dtype=int)
    if pulse_indices_arr.size:
        pulse_indices_arr = np.unique(pulse_indices_arr)

    if pulse_indices_arr.size:
        pulse_times_arr = t_values[pulse_indices_arr]
        pulse_saved_indices = np.clip(
            np.searchsorted(t_saved, pulse_times_arr), 0, len(t_saved) - 1
        )
    else:
        pulse_times_arr = np.array([], dtype=float)
        pulse_saved_indices = np.array([], dtype=int)

    if adaptive_on_decay:
        if pulse_indices_arr.size > 1:
            t_span = float(t_values[pulse_indices_arr[-1]] - t_values[pulse_indices_arr[0]])
            period = t_span / float(len(pulse_indices_arr) - 1) if t_span > 0.0 else np.inf
        else:
            period = np.inf
        seg_t_start = float(t_values[adaptive_window_start_idx]) if 0 <= adaptive_window_start_idx < n_t else float(t_values[0])
        seg_t_end = float(t_values[adaptive_window_end_idx]) if 0 <= adaptive_window_end_idx < n_t else float(t_values[-1])
        duration = max(seg_t_end - seg_t_start, 0.0)
        event_rate = float(pulse_indices_arr.size) / duration if pulse_indices_arr.size > 0 and duration > 0.0 else 0.0
        repetition_rate = float(pulse_indices_arr.size - 1) / duration if pulse_indices_arr.size > 1 and duration > 0.0 else 0.0
        pulse_segments_info = [{
            "t_start": seg_t_start,
            "t_end": seg_t_end,
            "count": int(pulse_indices_arr.size),
            "frequency": event_rate,
            "repetition_frequency": repetition_rate,
            "period": period,
            "mode": "on_decay",
            "probe_radius": adaptive_probe_radius,
            "min_gap_us": float(pulse_trigger_min_gap),
            "require_rise": bool(adaptive_require_rise),
        }]

    if pulse_time_window is None:
        freq_window_start = float(t_values[0])
        freq_window_end = float(t_values[-1])
    else:
        freq_window_start = max(float(t_values[0]), float(min(pulse_time_window)))
        freq_window_end = min(float(t_values[-1]), float(max(pulse_time_window)))
    if adaptive_on_decay and pulse_segments_info:
        freq_window_start = float(pulse_segments_info[0].get("t_start", freq_window_start))
        freq_window_end = float(pulse_segments_info[0].get("t_end", freq_window_end))

    freq_window_duration = max(freq_window_end - freq_window_start, 0.0)
    event_rate_hz = (
        float(pulse_indices_arr.size) / freq_window_duration * 1.0e6
        if pulse_indices_arr.size > 0 and freq_window_duration > 0.0
        else 0.0
    )
    if pulse_times_arr.size > 1:
        pulse_gaps = np.diff(pulse_times_arr)
        min_gap_us = float(np.min(pulse_gaps))
        mean_gap_us = float(np.mean(pulse_gaps))
        repetition_rate_hz = 1.0e6 / mean_gap_us if mean_gap_us > 0.0 else 0.0
    else:
        min_gap_us = np.inf
        mean_gap_us = np.inf
        repetition_rate_hz = None

    pulse_info = {
        "segments": pulse_segments_info,
        "count": int(pulse_indices_arr.size),
        "pulse_amount": float(pulse_amount),
        "source_model": "центральный гауссов профиль",
        "center_nm": float(config.pulse_center_nm),
        "sigma_nm": float(config.pulse_sigma_nm),
        "r_window": pulse_r_window,
        "indices": pulse_saved_indices,
        "pulse_times": pulse_times_arr,
        "time_window": pulse_time_window,
        "frequency_window": (freq_window_start, freq_window_end),
        "event_rate_hz": event_rate_hz,
        "repetition_rate_hz": repetition_rate_hz,
        "min_gap_us": min_gap_us,
        "mean_gap_us": mean_gap_us,
        "max_repetition_frequency_hz": float(config.pulse_max_frequency_hz),
        "max_frequency_hz": float(config.pulse_max_frequency_hz),
        "segment_default_mode": pulse_segment_default_mode,
        "trigger_mode": trigger_mode,
        "trigger_radius": adaptive_probe_radius,
        "trigger_min_gap": float(effective_min_gap_us),
        "trigger_slope_tol": float(pulse_trigger_slope_tol),
        "trigger_require_rise": bool(adaptive_require_rise),
        "outer_boundary_mode": boundary_mode,
        "geometry_mode": geometry_kind,
        "solver_convergence": {
            "total_steps": int(solver_total_steps),
            "converged_steps": int(solver_converged_steps),
            "failed_steps": int(solver_total_steps - solver_converged_steps),
            "failed_times": np.asarray(solver_failed_times, dtype=float),
            "residual_tol": float(config.SOLVER_RESIDUAL_TOL),
        },
    }
    return p_history, r_values, t_saved, D_values, Omega_values, pulse_info
