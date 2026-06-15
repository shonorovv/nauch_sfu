# -*- coding: utf-8 -*-
"""
scenarios/supervisor_revision.py - Сценарии для проверки замечаний научного руководителя.

Назначение:
1) воспроизвести графики, которые используются в исправленной курсовой;
2) проверить замечания научного руководителя:
   - связь условия инверсии с Pe и реальным неравенством Omega*p > F/r^2;
   - частоты 150-200 Гц;
   - сценарий без дрейфа;
   - сценарий умеренного контраста D;
   - сценарий без импульсной серии.

Запуск (из корня проекта):
    python -m scenarios.supervisor_revision

Результаты сохраняются в папку supervisor_revision_outputs.
"""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from solver.fp_solver import solve_fp_equation
from solver.geometry import radial_cell_measures


OUT_DIR = Path("supervisor_revision_outputs")
FIG_DIR = OUT_DIR / "figures"
DATA_DIR = OUT_DIR / "data"


# ---------------------------------------------------------------------------
# Временная замена параметров через config (setattr-паттерн)
# ---------------------------------------------------------------------------

@contextmanager
def temporary_global_settings(**kwargs):
    """
    Временно меняет глобальные параметры в config.py и возвращает их обратно.
    Используется для запуска сценария с отличными от дефолтных параметрами.
    """
    old_values = {name: getattr(config, name) for name in kwargs}
    try:
        for name, value in kwargs.items():
            setattr(config, name, value)
        yield
    finally:
        for name, value in old_values.items():
            setattr(config, name, value)


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def cell_measures(r: np.ndarray) -> np.ndarray:
    return radial_cell_measures(r, config.dr, "spherical")


def gradient_profile(p: np.ndarray, dr: float) -> np.ndarray:
    dp = np.empty_like(p)
    dp[1:-1] = (p[2:] - p[:-2]) / (2.0 * dr)
    dp[0] = (p[1] - p[0]) / dr
    dp[-1] = (p[-1] - p[-2]) / dr
    return dp


def contiguous_true_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Возвращает включительные интервалы индексов для True-участков."""
    regions: List[Tuple[int, int]] = []
    start = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        if (not value or idx == len(mask) - 1) and start is not None:
            end = idx if value and idx == len(mask) - 1 else idx - 1
            if end >= start:
                regions.append((start, end))
            start = None
    return regions


def nearest_time_index(t: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(t - float(target))))


# ---------------------------------------------------------------------------
# Диагностика инверсии
# ---------------------------------------------------------------------------

def inversion_diagnostics(
    r: np.ndarray,
    t: np.ndarray,
    p_hist: np.ndarray,
    zone: Tuple[float, float] = (300.0, 700.0),
    threshold: float = 1.05,
    relative_floor: float = 1e-5,
) -> Dict:
    """
    Диагностика инверсии через положительный градиент.

    I(t) считается не по всем парам точек подряд, а по положительным участкам профиля.
    Это устраняет ложные большие отношения на почти нулевых хвостах распределения.
    """
    zone_mask = (r >= zone[0]) & (r <= zone[1])
    r_zone = r[zone_mask]
    I = np.ones_like(t, dtype=float)
    left_r = np.full_like(t, np.nan, dtype=float)
    right_r = np.full_like(t, np.nan, dtype=float)
    max_dpdr = np.zeros_like(t, dtype=float)
    positive_width = np.zeros_like(t, dtype=float)

    for k in range(len(t)):
        p = p_hist[:, k]
        dp = gradient_profile(p, config.dr)
        p_zone = p[zone_mask]
        dp_zone = dp[zone_mask]
        max_dpdr[k] = float(np.max(dp_zone)) if dp_zone.size else 0.0
        floor = max(float(np.max(p)) * relative_floor, 1e-30)
        pos = (dp_zone > 0.0) & (p_zone > floor)
        best_ratio = 1.0
        best_pair = (np.nan, np.nan)
        best_width = 0.0
        for s, e in contiguous_true_regions(pos):
            if e <= s:
                continue
            local_p = p_zone[s:e+1]
            local_r = r_zone[s:e+1]
            left_idx_local = int(np.argmin(local_p))
            right_idx_local = int(np.argmax(local_p))
            if right_idx_local <= left_idx_local:
                continue
            denom = max(float(local_p[left_idx_local]), 1e-30)
            ratio = float(local_p[right_idx_local] / denom)
            if ratio > best_ratio:
                best_ratio = ratio
                best_pair = (float(local_r[left_idx_local]), float(local_r[right_idx_local]))
                best_width = float(local_r[right_idx_local] - local_r[left_idx_local])
        I[k] = best_ratio
        left_r[k], right_r[k] = best_pair
        positive_width[k] = best_width

    above = I >= threshold
    first_idx = int(np.argmax(above)) if np.any(above) else None
    max_idx = int(np.argmax(I)) if I.size else None
    return {
        "I": I,
        "left_r": left_r,
        "right_r": right_r,
        "max_dpdr": max_dpdr,
        "positive_width_nm": positive_width,
        "first_crossing": None if first_idx is None else {
            "time_us": float(t[first_idx]),
            "I": float(I[first_idx]),
            "left_r_nm": float(left_r[first_idx]),
            "right_r_nm": float(right_r[first_idx]),
        },
        "maximum": None if max_idx is None else {
            "time_us": float(t[max_idx]),
            "I": float(I[max_idx]),
            "left_r_nm": float(left_r[max_idx]),
            "right_r_nm": float(right_r[max_idx]),
            "max_dpdr": float(max_dpdr[max_idx]),
        },
        "threshold": float(threshold),
        "zone": [float(zone[0]), float(zone[1])],
    }


# ---------------------------------------------------------------------------
# Запуск одного сценария
# ---------------------------------------------------------------------------

def run_model_scenario(
    name: str,
    label: str,
    frequency_hz: Optional[float],
    D_pm: float,
    Omega_cleft: float,
    Omega_pm: float,
    num_pulses: int = 50,
    pulse_window: Tuple[float, float] = (0.0, 50000.0),
    t_max: float = 100000.0,
    single_pulse: bool = False,
) -> Dict:
    """Запускает один сценарий с параметрами, нужными для проверки научника."""
    if single_pulse:
        pulse_segments = None
        local_num_pulses = 1
        local_pulse_window = (0.0, 0.0)
        cap_frequency = 1e9
    elif frequency_hz is None:
        pulse_segments = None
        local_num_pulses = num_pulses
        local_pulse_window = pulse_window
        cap_frequency = 1e9
    else:
        pulse_segments = [{"t_start": pulse_window[0], "t_end": pulse_window[1], "mode": "hz", "value": float(frequency_hz)}]
        local_num_pulses = 0
        local_pulse_window = pulse_window
        cap_frequency = float(frequency_hz)

    with temporary_global_settings(
        geometry_mode="spherical",
        pulse_max_frequency_hz=cap_frequency,
        realtime_modeling=False,
    ):
        p_hist, r, t, D_values, Omega_values, pulse_info = solve_fp_equation(
            config.max_r,
            config.dt,
            config.dr,
            t_max,
            config.D_cleft,
            D_pm,
            config.xi_s,
            config.a,
            config.b,
            Omega_cleft,
            Omega_pm,
            config.pulse_amount,
            config.pulse_r_window,
            local_num_pulses,
            pulse_segments,
            local_pulse_window,
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

    diag = inversion_diagnostics(r, t, p_hist)
    Pe_out = float(Omega_pm * (config.b - config.a) / D_pm) if D_pm > 0 else math.inf
    Pe_in = float(Omega_cleft * (config.b - config.a) / config.D_cleft) if config.D_cleft > 0 else math.inf
    return {
        "name": name,
        "label": label,
        "frequency_hz": frequency_hz,
        "D_pm": float(D_pm),
        "Omega_cleft": float(Omega_cleft),
        "Omega_pm": float(Omega_pm),
        "Pe_in": Pe_in,
        "Pe_out": Pe_out,
        "r": r,
        "t": t,
        "p": p_hist,
        "D": D_values,
        "Omega": Omega_values,
        "pulse_info": pulse_info,
        "diagnostics": diag,
    }


# ---------------------------------------------------------------------------
# Графики
# ---------------------------------------------------------------------------

def plot_parameters(scenario: Dict, path: Path) -> None:
    r = scenario["r"]
    D = scenario["D"]
    Omega = scenario["Omega"]
    L = config.b - config.a
    Pe = Omega * L / np.maximum(D, 1e-30)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.6), sharex=True)
    axes[0].plot(r, D, lw=2)
    axes[0].set_ylabel("D, нм²/мкс")
    axes[1].plot(r, Omega, lw=2)
    axes[1].set_ylabel("Ω, нм/мкс")
    axes[2].plot(r, Pe, lw=2)
    axes[2].axhline(1.0, ls="--", lw=1, color="black")
    axes[2].set_ylabel("Pe=ΩL/D")
    axes[2].set_xlabel("Радиус r, нм")
    for ax in axes:
        ax.axvspan(config.a, config.b, alpha=0.12)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Радиально-зависимые параметры переноса")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_condition(scenario: Dict, path: Path) -> None:
    r = scenario["r"]
    t = scenario["t"]
    p = scenario["p"]
    D = scenario["D"]
    Omega = scenario["Omega"]
    max_time = scenario["diagnostics"]["maximum"]["time_us"] if scenario["diagnostics"]["maximum"] else 50000.0
    idx = nearest_time_index(t, max_time)
    dp = gradient_profile(p[:, idx], config.dr)
    J = -D * dp + Omega * p[:, idx]
    left = Omega * p[:, idx]
    right = J
    margin = left - right
    mask = (r >= 300.0) & (r <= 700.0)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.4), sharex=True)
    axes[0].plot(r[mask], left[mask], lw=2, label="Ωp")
    axes[0].plot(r[mask], right[mask], lw=2, label="F/r² = J")
    axes[0].set_ylabel("члены условия")
    axes[0].set_title(f"Проверка условия инверсии при t={t[idx]:.0f} мкс")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(r[mask], margin[mask], lw=2, label="Ωp - F/r²")
    axes[1].axhline(0.0, ls="--", color="black", lw=1)
    axes[1].fill_between(r[mask], 0.0, margin[mask], where=margin[mask] > 0, alpha=0.22)
    axes[1].set_xlabel("Радиус r, нм")
    axes[1].set_ylabel("разность")
    axes[1].set_title("Положительная область соответствует ∂p/∂r > 0")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_zoom_profiles_and_derivative(scenario: Dict, profile_path: Path, deriv_path: Path) -> None:
    r = scenario["r"]
    t = scenario["t"]
    p = scenario["p"]
    diag = scenario["diagnostics"]
    key_times = [50000.0]
    if diag["first_crossing"]:
        key_times.append(diag["first_crossing"]["time_us"])
    if diag["maximum"]:
        key_times.append(diag["maximum"]["time_us"])
    key_times.append(60000.0)
    unique_times = []
    for value in key_times:
        if not any(abs(value - old) < 0.5 * config.dt for old in unique_times):
            unique_times.append(value)
    idxs = [nearest_time_index(t, x) for x in unique_times]
    mask = (r >= 300.0) & (r <= 700.0)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for idx in idxs:
        ax.plot(r[mask], p[mask, idx], lw=2, label=f"t={t[idx]:.0f} мкс")
    if diag["maximum"]:
        ax.axvline(diag["maximum"]["left_r_nm"], ls="--", lw=1, color="black", alpha=0.5)
        ax.axvline(diag["maximum"]["right_r_nm"], ls="--", lw=1, color="black", alpha=0.5)
    ax.set_xlabel("Радиус r, нм")
    ax.set_ylabel("p(r,t)")
    ax.set_title("Участок смены наклона профиля")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(profile_path, dpi=220)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for idx in idxs:
        dp = gradient_profile(p[:, idx], config.dr)
        ax.plot(r[mask], dp[mask], lw=2, label=f"t={t[idx]:.0f} мкс")
    ax.axhline(0.0, ls="--", color="black", lw=1)
    ax.set_xlabel("Радиус r, нм")
    ax.set_ylabel("∂p/∂r")
    ax.set_title("Производная профиля в области инверсии")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(deriv_path, dpi=220)
    plt.close(fig)


def plot_inversion_index(scenario: Dict, path: Path) -> None:
    t = scenario["t"]
    I = scenario["diagnostics"]["I"]
    threshold = scenario["diagnostics"]["threshold"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(t, I, lw=2)
    ax.axhline(threshold, ls="--", color="black", lw=1, label="порог 1.05")
    first = scenario["diagnostics"]["first_crossing"]
    maximum = scenario["diagnostics"]["maximum"]
    if first:
        ax.axvline(first["time_us"], ls=":", lw=1.2, label="первое превышение")
    if maximum:
        ax.axvline(maximum["time_us"], ls="-.", lw=1.2, label="максимум индекса")
    ax.set_xlim(0.0, min(float(t[-1]), 80000.0))
    ax.set_xlabel("Время t, мкс")
    ax.set_ylabel("I(t)")
    ax.set_title("Индекс инверсии во времени")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_scenario_comparison(scenarios: List[Dict], path: Path) -> None:
    labels = [s["label"] for s in scenarios]
    max_I = [float(np.max(s["diagnostics"]["I"])) for s in scenarios]
    crossed = [1.0 if s["diagnostics"]["first_crossing"] else 0.0 for s in scenarios]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, max_I)
    ax.axhline(1.05, ls="--", color="black", lw=1, label="порог 1.05")
    for bar, flag in zip(bars, crossed):
        text = "да" if flag else "нет"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), text,
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("max I(t)")
    ax.set_title("Проверка устойчивости инверсии в разных сценариях")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_summary(scenarios: List[Dict], path: Path) -> None:
    rows = []
    for s in scenarios:
        diag = s["diagnostics"]
        rows.append({
            "name": s["name"],
            "label": s["label"],
            "frequency_hz": s["frequency_hz"],
            "pulse_count": int(s["pulse_info"].get("count", 0)),
            "event_rate_hz": float(s["pulse_info"].get("event_rate_hz", 0.0)),
            "repetition_rate_hz": s["pulse_info"].get("repetition_rate_hz", None),
            "D_pm": s["D_pm"],
            "Omega_cleft": s["Omega_cleft"],
            "Omega_pm": s["Omega_pm"],
            "Pe_in": s["Pe_in"],
            "Pe_out": s["Pe_out"],
            "first_crossing": diag["first_crossing"],
            "maximum": diag["maximum"],
            "max_I": float(np.max(diag["I"])),
        })
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = [
        run_model_scenario("baseline_1000hz", "1000 Гц, базовый", 1000.0, D_pm=0.5, Omega_cleft=0.05, Omega_pm=0.15),
        run_model_scenario("phys_200hz",       "200 Гц",            200.0,  D_pm=0.5, Omega_cleft=0.05, Omega_pm=0.15),
        run_model_scenario("phys_150hz",       "150 Гц",            150.0,  D_pm=0.5, Omega_cleft=0.05, Omega_pm=0.15),
        run_model_scenario("no_drift",         "без дрейфа",        1000.0, D_pm=0.5, Omega_cleft=0.0,  Omega_pm=0.0),
        run_model_scenario("moderate_D",       "умеренный контраст D", 1000.0, D_pm=5.0, Omega_cleft=0.05, Omega_pm=0.15),
        run_model_scenario("single_pulse",     "один импульс",      None,   D_pm=0.5, Omega_cleft=0.05, Omega_pm=0.15, single_pulse=True),
    ]

    baseline = scenarios[0]
    plot_parameters(baseline, FIG_DIR / "01_parameters_pe.png")
    plot_condition(baseline, FIG_DIR / "02_condition_inversion.png")
    plot_zoom_profiles_and_derivative(baseline, FIG_DIR / "03_zoom_profiles.png", FIG_DIR / "04_derivative.png")
    plot_inversion_index(baseline, FIG_DIR / "05_inversion_index.png")
    plot_scenario_comparison(scenarios, FIG_DIR / "06_scenario_comparison.png")
    save_summary(scenarios, DATA_DIR / "scenario_summary.json")

    # Сохраняем базовые массивы, чтобы графики можно было перепостроить.
    np.savez_compressed(
        DATA_DIR / "baseline_arrays.npz",
        r=baseline["r"],
        t=baseline["t"],
        p=baseline["p"],
        D=baseline["D"],
        Omega=baseline["Omega"],
        I=baseline["diagnostics"]["I"],
    )

    print(f"Готово. Результаты сохранены в {OUT_DIR.resolve()}")
    for s in scenarios:
        first = s["diagnostics"]["first_crossing"]
        max_item = s["diagnostics"]["maximum"]
        print("-", s["label"], "max I=", f"{np.max(s['diagnostics']['I']):.3f}", "first=", first, "max=", max_item)


if __name__ == "__main__":
    main()
