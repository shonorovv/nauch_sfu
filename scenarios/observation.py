# -*- coding: utf-8 -*-
"""
scenarios/observation.py - модели наблюдения: как концентрация глутамата c(r, t)
превращается в то, что регистрировал прибор в эксперименте.

1. NMDA-рецептор с конкурентным антагонистом (Clements et al. 1992, Fig. 3A).
   Состояния: R, AR, A2R, CR, C2R, ACR (A - глутамат, C - антагонист).
   Константы связывания из Fig. 3A (мкМ^-1 с^-1 и с^-1):
       R  -> AR   10 (2 сайта × 5)     AR  -> R    5
       AR -> A2R   5                   A2R -> AR  10 (2 × 5)
       R  -> CR   2*kon_C              CR  -> R   koff_C
       AR -> ACR  kon_C                ACR -> AR  koff_C
       CR -> ACR   5                   ACR -> CR   5
       CR -> C2R  kon_C                C2R -> CR  2*koff_C
   D-AA: kon_C = 7, koff_C = 210 (Kd 30 мкМ; Clements 1992, ссылка 23).
   D-CPP: Kd = 0,4 мкМ, koff_C = 1,1 с^-1 (текст Clements 1992) -> kon_C = 2,75.
   Переходы A2R -> открыто/десенситизация (ro, rc, rd, rr) в статье не даны числами
   и здесь не моделируются: амплитуда EPSC берётся пропорциональной максимуму
   A2R, т.е. числу рецепторов, связавших две молекулы глутамата. Это допущение
   проверяется в validate_clements(): с импульсом самих авторов (1,1 мМ, 1,2 мс)
   модель должна воспроизвести их пунктир на Fig. 3B.
   Антагонист до выброса находится в равновесии с рецепторами (5 с аппликации).

2. Датчик iGluSnFr (Matthews et al. 2022): dB/dt = kon*c*(1-B) - koff*B.

3. AMPA-рецептор: схема и константы Jonas, Major & Sakmann 1993 (J Physiol 472:615),
   числа взяты из Методов Franks, Stevens & Sejnowski 2003 (J Neurosci 23:3186),
   где эта схема воспроизведена текстом. Состояния C0, C1, C2, O, C3, C4, C5:
       C0 -> C1  4,59·10^6 М^-1с^-1 ·A      C1 -> C0  4260 с^-1
       C1 -> C2  2,84·10^7 М^-1с^-1 ·A      C2 -> C1  3260
       C2 -> O   4240                        O  -> C2  900
       C1 -> C3  2890                        C3 -> C1  39,2
       C3 -> C4  1,27·10^6 М^-1с^-1 ·A      C4 -> C3  45,7
       C2 -> C4  172                         C4 -> C2  0,727
       C4 -> C5  16,8                        C5 -> C4  190,4
       O  -> C5  17,7                        C5 -> O   4
   Амплитуда AMPA-тока ∝ максимуму доли открытых каналов O(t).
"""
import numpy as np

GLU = {"k1": 10.0, "km1": 5.0, "k2": 5.0, "km2": 10.0, "kCA": 5.0, "kmCA": 5.0}   # 1/(мкМ*с), 1/с
ANTAG = {"D-AA": {"kon": 7.0, "koff": 210.0}, "D-CPP": {"kon": 1.1 / 0.4, "koff": 1.1}}


def _rhs(x, A, C, ka, kd):
    """x: (6, n) = R, AR, A2R, CR, C2R, ACR; A: (n,) глутамат мкМ; C: антагонист мкМ. Возвращает dx/dt, 1/с."""
    R, AR, A2R, CR, C2R, ACR = x
    g = GLU
    f_R_AR = g["k1"] * A * R - g["km1"] * AR
    f_AR_A2R = g["k2"] * A * AR - g["km2"] * A2R
    f_R_CR = 2 * ka * C * R - kd * CR
    f_AR_ACR = ka * C * AR - kd * ACR
    f_CR_ACR = g["kCA"] * A * CR - g["kmCA"] * ACR
    f_CR_C2R = ka * C * CR - 2 * kd * C2R
    dR = -f_R_AR - f_R_CR
    dAR = f_R_AR - f_AR_A2R - f_AR_ACR
    dA2R = f_AR_A2R
    dCR = f_R_CR - f_CR_ACR - f_CR_C2R
    dC2R = f_CR_C2R
    dACR = f_AR_ACR + f_CR_ACR
    return np.array([dR, dAR, dA2R, dCR, dC2R, dACR])


def a2r_peak(A_t_uM, t_us, C_uM, antagonist="D-AA", weights=None, dt_us=1.0):
    """
    Пик доли A2R (усреднённой по рецепторам с весами weights) при концентрациях
    антагониста C_uM (число или массив). A_t_uM: (n_nodes, n_t) глутамат у рецепторов
    в мкМ, t_us: (n_t,) моменты, мкс. RK4 с шагом dt_us, все концентрации считаются
    одновременно (векторизация: x имеет форму (6, n_nodes, n_conc)).
    Возвращает массив пиков той же длины, что C_uM.
    """
    A_t_uM = np.atleast_2d(A_t_uM)
    C = np.atleast_1d(np.asarray(C_uM, float))[None, :]          # (1, m)
    n = A_t_uM.shape[0]
    w = np.ones(n) / n if weights is None else np.asarray(weights, float) / np.sum(weights)
    ka, kd = ANTAG[antagonist]["kon"], ANTAG[antagonist]["koff"]
    q = C / (kd / ka)
    Z = (1 + q) ** 2
    x = np.zeros((6, n, C.shape[1]))
    x[0] = 1 / Z; x[3] = 2 * q / Z; x[4] = q * q / Z      # равновесие с антагонистом до выброса
    tgrid = np.arange(t_us[0], t_us[-1] + 1e-9, dt_us)
    Ag = np.array([np.interp(tgrid, t_us, A_t_uM[i]) for i in range(n)])   # (n, nt)
    h = dt_us * 1e-6
    best = np.zeros(C.shape[1])
    for k in range(tgrid.size - 1):
        A0 = Ag[:, k, None]; A1 = Ag[:, k + 1, None]; Am = 0.5 * (A0 + A1)
        k1 = _rhs(x, A0, C, ka, kd)
        k2 = _rhs(x + 0.5 * h * k1, Am, C, ka, kd)
        k3 = _rhs(x + 0.5 * h * k2, Am, C, ka, kd)
        k4 = _rhs(x + h * k3, A1, C, ka, kd)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        np.maximum(best, w @ x[2], out=best)
    return best


def epsc_ratio(A_t_uM, t_us, concentrations_uM, antagonist="D-AA", weights=None, dt_us=1.0):
    """Нормированный EPSC = пик A2R с антагонистом / пик A2R без него (контроль C = 0)."""
    C = np.concatenate([[0.0], np.asarray(concentrations_uM, float)])
    peaks = a2r_peak(A_t_uM, t_us, C, antagonist, weights, dt_us)
    return peaks[1:] / peaks[0]


def a2r_peak_per_node(A_t_uM, t_us, dt_us=5.0):
    """
    Пик доли A2R без антагониста отдельно для каждого узла (рецептор в точке r).
    Нужен для Matthews 2022 Fig. 3C: NMDA-uEPSC на шипике в зависимости от
    расстояния до точки анкейджинга. Схема и константы те же, что выше (Clements 1992).
    """
    A_t_uM = np.atleast_2d(A_t_uM)
    n = A_t_uM.shape[0]
    x = np.zeros((6, n, 1)); x[0] = 1.0
    C = np.zeros((1, 1))
    tgrid = np.arange(t_us[0], t_us[-1] + 1e-9, dt_us)
    Ag = np.array([np.interp(tgrid, t_us, A_t_uM[i]) for i in range(n)])
    h = dt_us * 1e-6
    best = np.zeros(n)
    for k in range(tgrid.size - 1):
        A0 = Ag[:, k, None]; A1 = Ag[:, k + 1, None]; Am = 0.5 * (A0 + A1)
        k1 = _rhs(x, A0, C, 1.0, 1.0)
        k2 = _rhs(x + 0.5 * h * k1, Am, C, 1.0, 1.0)
        k3 = _rhs(x + 0.5 * h * k2, Am, C, 1.0, 1.0)
        k4 = _rhs(x + h * k3, A1, C, 1.0, 1.0)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        np.maximum(best, x[2, :, 0], out=best)
    return best


AMPA = {  # в мкМ^-1 с^-1 и с^-1 (1 М^-1 = 1e-6 мкМ^-1)
    "c0c1": 4.59, "c1c0": 4260.0, "c1c2": 28.4, "c2c1": 3260.0, "c2o": 4240.0, "oc2": 900.0,
    "c1c3": 2890.0, "c3c1": 39.2, "c3c4": 1.27, "c4c3": 45.7, "c2c4": 172.0, "c4c2": 0.727,
    "c4c5": 16.8, "c5c4": 190.4, "oc5": 17.7, "c5o": 4.0,
}


def _rhs_ampa(x, A):
    """x: (7, n) = C0, C1, C2, O, C3, C4, C5; A: (n,) глутамат, мкМ. Возвращает dx/dt, 1/с."""
    k = AMPA
    C0, C1, C2, O, C3, C4, C5 = x
    f01 = k["c0c1"] * A * C0 - k["c1c0"] * C1
    f12 = k["c1c2"] * A * C1 - k["c2c1"] * C2
    f2o = k["c2o"] * C2 - k["oc2"] * O
    f13 = k["c1c3"] * C1 - k["c3c1"] * C3
    f34 = k["c3c4"] * A * C3 - k["c4c3"] * C4
    f24 = k["c2c4"] * C2 - k["c4c2"] * C4
    f45 = k["c4c5"] * C4 - k["c5c4"] * C5
    fo5 = k["oc5"] * O - k["c5o"] * C5
    return np.array([-f01, f01 - f12 - f13, f12 - f2o - f24, f2o - fo5, f13 - f34, f34 + f24 - f45, f45 + fo5])


def ampa_open_peak_per_node(A_t_uM, t_us, dt_us=2.0, t_end_us=30000.0):
    """
    Пик доли открытых AMPA-каналов O(t) отдельно для каждого узла (Matthews 2022 Fig. 4C, 4D).
    Интегрирование RK4 до t_end_us (пик AMPA-uEPSC в эксперименте наступает за 3-4 мс).
    """
    A_t_uM = np.atleast_2d(A_t_uM)
    n = A_t_uM.shape[0]
    x = np.zeros((7, n)); x[0] = 1.0
    t_end = min(t_us[-1], t_end_us)
    tgrid = np.arange(t_us[0], t_end + 1e-9, dt_us)
    Ag = np.array([np.interp(tgrid, t_us, A_t_uM[i]) for i in range(n)])
    h = dt_us * 1e-6
    best = np.zeros(n)
    for k in range(tgrid.size - 1):
        A0 = Ag[:, k]; A1 = Ag[:, k + 1]; Am = 0.5 * (A0 + A1)
        k1 = _rhs_ampa(x, A0)
        k2 = _rhs_ampa(x + 0.5 * h * k1, Am)
        k3 = _rhs_ampa(x + 0.5 * h * k2, Am)
        k4 = _rhs_ampa(x + h * k3, A1)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        np.maximum(best, x[3], out=best)
    return best


def sensor_peak(c_uM, t_us, Kd_uM=4.9, tau_off_ms=60.0):
    """Пик занятости датчика B(r) за всё время (точное решение на каждом шаге)."""
    koff = 1.0 / (tau_off_ms * 1000.0); kon = koff / Kd_uM
    B = np.zeros(c_uM.shape[0]); Bmax = np.zeros_like(B)
    for k in range(1, t_us.size):
        h = t_us[k] - t_us[k - 1]; C = 0.5 * (c_uM[:, k] + c_uM[:, k - 1])
        a = kon * C + koff; Binf = kon * C / a
        B = Binf + (B - Binf) * np.exp(-a * h); np.maximum(Bmax, B, out=Bmax)
    return Bmax
