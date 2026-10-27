# -*- coding: utf-8 -*-
"""
analysis/excitotoxicity.py - СОВМЕСТИМОСТЬ со старым именем. Не использовать в новом коде.

Функция измеряет МАССУ глутамата за радиусом и её интеграл по времени, то есть внесинаптическую
экспозицию (analysis/extrasynaptic_exposure.py, compute_mass_exposure). Это не рецепторная
активация и не эксайтотоксичность (P0.5 ТЗ от 13.10.2026): физическая эксайтотоксичность
требует кинетики NMDA-R и модели Ca2+.

Параметр concentration_threshold оставлен ради старых вызовов, но по смыслу это ПОРОГ МАССЫ
(в единицах M_extra), а не концентрации в мкМ. Порог в мкМ применяется к локальной концентрации:
compute_local_concentration_exposure.
"""
from analysis.extrasynaptic_exposure import compute_mass_exposure


def compute_extrasynaptic_exposure(
    r_values,
    t_values,
    shell_probabilities,
    extrasynaptic_radius,
    concentration_threshold=None,
):
    """Обёртка над compute_mass_exposure; ключи результата прежние + явные имена порога массы."""
    res = compute_mass_exposure(
        r_values, t_values, shell_probabilities, extrasynaptic_radius,
        mass_threshold=concentration_threshold,
    )
    if res is None:
        return None
    res["extrasynaptic_radius"] = res["r_edge_nm"]
    if concentration_threshold is not None:
        res["concentration_threshold"] = res["mass_threshold"]          # устаревшее имя: это порог массы
        res["time_above_threshold"] = res["time_above_mass_threshold"]
        res["first_time_above_threshold"] = res["first_time_above_mass_threshold"]
    return res
