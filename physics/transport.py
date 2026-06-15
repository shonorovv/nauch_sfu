# -*- coding: utf-8 -*-
"""
physics/transport.py - Коэффициенты переноса: диффузия D(r) и дрейф Omega(r).

Все функции принимают параметры явно, глобальных переменных не читают.
"""

import numpy as np


def poly_transition_weight(r_values, xi_s, a, b):
    """Полиномиальная функция перехода от 0 (при r<=a) до 1 (при r>=b)."""
    w_values = np.zeros_like(r_values)
    w_values[r_values > b] = 1.0
    mask_mid = (r_values >= a) & (r_values <= b)
    if np.any(mask_mid):
        r_mid = r_values[mask_mid]
        powers = np.vstack([r_mid**i for i in range(len(xi_s))])
        w_values[mask_mid] = np.dot(xi_s, powers)
    return w_values


def sigmoid_transition_weight(r_values, a, b, steepness):
    """Сигмоидальная функция перехода от 0 до 1 в зоне [a, b]."""
    span = max(abs(b - a), 1e-12)
    steepness = max(abs(steepness), 1e-12)
    width = span / steepness
    center = 0.5 * (a + b)
    x = (r_values - center) / max(width, 1e-12)
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def transition_weight(r_values, kind, xi_s, a, b, steepness):
    """Выбирает тип перехода: 'poly' или 'sigmoid'."""
    if kind == "poly":
        return poly_transition_weight(r_values, xi_s, a, b)
    if kind == "sigmoid":
        return sigmoid_transition_weight(r_values, a, b, steepness)
    raise ValueError(f"Неизвестный тип перехода: {kind}")


def D_variable(r_values, D_cleft, D_pm, xi_s, a, b, transition_kind, transition_steepness):
    """
    Радиально-зависимый коэффициент диффузии D(r).

    D(r) = D_cleft внутри синапса (r < a),
    D(r) = D_pm    снаружи синапса (r > b),
    плавный переход на [a, b].
    """
    weight = transition_weight(r_values, transition_kind, xi_s, a, b, transition_steepness)
    return D_cleft + weight * (D_pm - D_cleft)


def Omega_variable(r_values, Omega_cleft, Omega_pm, xi_s, a, b, transition_kind, transition_steepness):
    """
    Радиально-зависимый коэффициент дрейфа Omega(r).

    Чтобы отключить дрейф: Omega_cleft=0, Omega_pm=0.
    """
    weight = transition_weight(r_values, transition_kind, xi_s, a, b, transition_steepness)
    return Omega_cleft + weight * (Omega_pm - Omega_cleft)


def find_transition_start_radius(r_values, D_values, D_cleft, D_pm, fraction=0.05):
    """Ищет радиус, где D(r) начинает заметно отклоняться от D_cleft."""
    if D_values.size == 0:
        return None
    delta = D_pm - D_cleft
    if abs(delta) < 1e-12:
        return None
    threshold = abs(delta) * max(float(fraction), 0.0)
    diff = np.abs(D_values - D_cleft)
    mask = diff >= threshold
    if not np.any(mask):
        return None
    idx = int(np.argmax(mask))
    return float(r_values[idx])
