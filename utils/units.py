# -*- coding: utf-8 -*-
"""
utils/units.py - Перевод величин решателя в физические единицы (мкМ).

ЗАЧЕМ
-----
Решатель (solver/fp_solver.py) работает в "своих" единицах:
    r - нм, t - мкс, D - нм²/мкс,
    c - "масса на радиальную меру ячейки".

Радиальная мера ячейки в solver/geometry.py берётся БЕЗ множителя 4π
(сфера) или 2π (цилиндр). Поэтому если в режиме
normalization_mode = "concentration" импульс добавляет N молекул
(pulse_amount = N), то:

    сфера:   c_истинная [молекул/нм³] = c_решателя / (4π)
    цилиндр: c_истинная [молекул/нм³] = c_решателя / (2π · h),
             где h - высота щели (цилиндр = тонкий диск высоты h).

Дальше учитываем, что медиатор живёт только во внеклеточном
пространстве, а оно занимает долю α объёма ткани (extracellular
volume fraction, ~0.2 по Zheng et al., 2008):

    c_во_внеклеточной_жидкости = c_истинная / α

И переводим молекулы/нм³ в мкМ:
    1 нм³ = 1e-24 л  ->  1 молекула/нм³ = 1e24 / N_A моль/л ≈ 1.66 М
    -> множитель 1e24 / N_A * 1e6 ≈ 1.66e6 мкМ на 1 молекулу/нм³.

ГДЕ МОЖНО ОШИБИТЬСЯ
-------------------
1) Переводить в мкМ имеет смысл ТОЛЬКО в режиме "concentration".
   В режиме "probability" масса принудительно = 1, абсолютной
   концентрации там нет.
2) α одинакова во всей области - это упрощение. В реальной щели
   медиатор находится в тонком диске (~20 нм), а сферическая
   геометрия "размазывает" его по шару радиуса a. Поэтому
   концентрация внутри синапса в сферической модели занижена.
3) Модуль специально НЕ импортирует config (чтобы config мог
   импортировать его без циклического импорта). Все параметры
   передаются явно.
"""

import math

import numpy as np

AVOGADRO = 6.02214076e23
# 1 молекула/нм³ в мкМ: 1e24 нм³ в литре / N_A -> моль/л, *1e6 -> мкМ
MOLECULES_PER_NM3_TO_UM = 1.0e24 / AVOGADRO * 1.0e6  # ≈ 1.6605e6


def geometry_factor(geometry_mode="spherical", cleft_height_nm=20.0):
    """
    Множитель, который превращает радиальную меру решателя в реальный объём.
    сфера: 4π; цилиндр (диск высоты h): 2π·h.
    """
    mode = str(geometry_mode).lower()
    if mode in ("spherical", "sphere", "3d"):
        return 4.0 * math.pi
    if mode in ("cylindrical", "radial", "polar", "2d"):
        return 2.0 * math.pi * float(cleft_height_nm)
    raise ValueError(f"Неизвестная геометрия: {geometry_mode}")


def solver_to_uM_factor(alpha, geometry_mode="spherical", cleft_height_nm=20.0):
    """Множитель: c_мкМ = c_решателя * factor."""
    alpha = float(alpha)
    if alpha <= 0.0:
        raise ValueError("Доля внеклеточного объёма α должна быть > 0")
    return MOLECULES_PER_NM3_TO_UM / (geometry_factor(geometry_mode, cleft_height_nm) * alpha)


def solver_to_uM(c_solver, alpha, geometry_mode="spherical", cleft_height_nm=20.0):
    """Переводит массив c(r,t) из единиц решателя в мкМ."""
    return np.asarray(c_solver, dtype=float) * solver_to_uM_factor(alpha, geometry_mode, cleft_height_nm)


def uM_to_solver(c_uM, alpha, geometry_mode="spherical", cleft_height_nm=20.0):
    """Обратный перевод: мкМ -> единицы решателя (нужно для Km и Vmax)."""
    return np.asarray(c_uM, dtype=float) / solver_to_uM_factor(alpha, geometry_mode, cleft_height_nm)


def uptake_to_solver_units(Km_uM, Vmax_cleft_uM_per_us, Vmax_pm_uM_per_us,
                           alpha, geometry_mode="spherical", cleft_height_nm=20.0):
    """
    Переводит параметры Михаэлиса-Ментен из мкМ и мкМ/мкс в единицы решателя.

    Сток в решателе: -Vmax*c/(Km+c). Km и Vmax обязаны быть в тех же
    единицах, что и c, поэтому оба делятся на один и тот же множитель.
    Возвращает (Km_solver, Vmax_cleft_solver, Vmax_pm_solver).
    """
    f = solver_to_uM_factor(alpha, geometry_mode, cleft_height_nm)
    return float(Km_uM) / f, float(Vmax_cleft_uM_per_us) / f, float(Vmax_pm_uM_per_us) / f
