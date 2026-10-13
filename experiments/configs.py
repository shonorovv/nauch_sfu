# -*- coding: utf-8 -*-
"""
experiments/configs.py - именованные конфигурации сценариев (ТЗ 13.10.2026, этап 0 п. 4).

Каждая конфигурация - полный набор физических параметров. Сценарий передаёт его
решателю целиком (experiments/run_stage02.py), не полагаясь на глобальные значения
config.py и на порядок импорта модулей.

Единицы: длины - нм, время - мкс, D - нм²/мкс, Ω - нм/мкс, Vmax - мкМ/мкс, Km - мкМ.
"""
import copy

# Историческая модель курсовой: значения config.py на коммите c05e886 (09.10.2026).
# Математический пример. Физиологически не верифицирована.
LEGACY_COURSEWORK = {
    "name": "legacy_coursework",
    "label": "исторический сценарий курсовой (config.py)",
    "geometry": "spherical",
    "a_nm": 200.0, "b_nm": 400.0, "h_nm": 20.0, "alpha": 0.2,
    "D_cleft": 10.0, "D_pm": 0.5,
    "Omega_cleft": 0.05, "Omega_pm": 0.15,
    "uptake": "none", "Vmax_cleft": 0.0, "Vmax_pm": 0.0, "Km": 20.0,
    "uptake_scheme": "imex", "clip_negative": True,
    "N": 3000.0, "sigma_nm": 500.0, "window_nm": 500.0,
    "R_nm": 1000.0, "dr_nm": 0.5, "dt_us": 0.5, "t_max_us": 1000.0,
    "boundary": "dirichlet",
    "note": "num_pulses = 100 в config.py, фактически 1 (предел 100 Гц в окне 0-500 мкс); "
            "N = 3000 добавлено, чтобы перевести вероятность в концентрацию",
}

# Физическая ветка, гиппокамп: геометрия Barbour 2001, Ω = 0, без захвата (контроль (а) ТЗ).
HIPPOCAMPUS_NO_UPTAKE = {
    "name": "baseline_physical",
    "label": "Barbour-геометрия, без захвата и дрейфа",
    "geometry": "barbour_mixed",
    "a_nm": 200.0, "b_nm": 400.0, "h_nm": 20.0, "alpha": 0.2,     # Barbour 2001: h 20 нм, α 0,2, переход 200 нм
    "D_cleft": 330.0,     # 0,33 мкм²/мс: Lotter et al. 2021 (по Nielsen 2004); у Barbour D_cleft = D_free = 760
    "D_pm": 320.0,        # 0,32 мкм²/мс: Zheng et al. 2008, нейропиль CA1
    "Omega_cleft": 0.0, "Omega_pm": 0.0,
    "uptake": "none", "Vmax_cleft": 0.0, "Vmax_pm": 0.0, "Km": 20.0,
    "uptake_scheme": "split_implicit", "clip_negative": False,
    "N": 3000.0, "sigma_nm": 20.0, "window_nm": 100.0,
    "R_nm": 6000.0, "dr_nm": 2.0, "dt_us": 0.5, "t_max_us": 5000.0,
    "boundary": "dirichlet",
    "note": "N = 3000: Zheng 2008, Lotter 2021",
}

# То же с насыщаемым захватом (контроль (б) ТЗ). Km и Vmax - рабочие оценки, не измерения.
HIPPOCAMPUS_WITH_UPTAKE = dict(copy.deepcopy(HIPPOCAMPUS_NO_UPTAKE), **{
    "name": "with_uptake",
    "label": "Barbour-геометрия, насыщаемый захват, без дрейфа",
    "uptake": "saturable",
    "Vmax_pm": 3.6e-3,    # пересчёт поверхностной константы глии Lotter 2021: κ·(площадь/объём)·Km
    "note": "Vmax_pm и Km - рабочие оценки (см. docs/stage02_report.md), не константы EAAT",
})

# Исследовательский: дрейф (только с явной пометкой, ТЗ раздел 2).
EXPLORATORY_DRIFT = dict(copy.deepcopy(HIPPOCAMPUS_WITH_UPTAKE), **{
    "name": "exploratory_drift",
    "label": "исследовательский: дрейф Ω (не физиологический механизм)",
    "Omega_cleft": 0.05, "Omega_pm": 0.15,
})

CONFIGS = {c["name"]: c for c in (LEGACY_COURSEWORK, HIPPOCAMPUS_NO_UPTAKE, HIPPOCAMPUS_WITH_UPTAKE, EXPLORATORY_DRIFT)}


def get(name):
    """Глубокая копия конфигурации по имени."""
    return copy.deepcopy(CONFIGS[name])
