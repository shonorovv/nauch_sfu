# -*- coding: utf-8 -*-
"""
config.py - Все параметры модели.

Чтобы запустить без импульсов: num_pulses = 0
Чтобы запустить без дрейфа:    Omega_cleft = Omega_pm = 0.0
Чтобы сменить геометрию:       geometry_mode = "cylindrical"
Чтобы отключить анимацию:      enable_animation = False
Чтобы отключить экспорт:       export_figures = export_data = False
"""

import os
import numpy as np
import numpy.linalg as lin
import matplotlib
from pathlib import Path

# --- Настройка matplotlib до первого import pyplot ---
if os.name == "nt" and os.environ.get("MPLBACKEND", "").lower() == "agg":
    os.environ.pop("MPLBACKEND", None)
if os.environ.get("MPLBACKEND") is None and os.name != "nt" and not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

# =============================================================================
# Численные константы (не трогать без причины)
# =============================================================================
TIME_TOL = 1e-12
EPS = 1e-30
VERBOSE = False
SOLVER_RESIDUAL_TOL = 1e-8
# Схема безусловно устойчива (неявный Эйлер), поэтому невязку не обязательно
# проверять на каждом шаге - каждая проверка стоит 2-3 синхронизации
# GPU<->CPU. При биологически реалистичном t_max (десятки-сотни тысяч
# шагов) это доминирует во времени счёта. 1 = проверять каждый шаг (как
# было раньше, нужно для точной диагностики); для длинных реалистичных
# прогонов достаточно проверять раз в 25-50 шагов.
solver_residual_check_stride = 25

use_gpu = True              # True = GPU (CuPy), False = CPU (NumPy/scipy)
max_saved_frames = 2000     # максимум кадров p(r,t), хранимых в памяти

# =============================================================================
# [РУЧНОЕ ЗАПОЛНЕНИЕ] Параметры модели - диффузия
# =============================================================================
# D_cleft=10, D_pm=0.5 - значения из курсовой (Table 1/2), не из статьи
# Shuvaev et al. 2024 (D_cleft=760, D_pm=297). У курсовой другая, сферическая
# геометрия (собственное расширение модели Barbour, не опечатка) и именно
# при этих D вместе с Omega_cleft=0.05/Omega_pm=0.15 получается число Пекле
# Pe~1 в центре и Pe~60 снаружи - баланс, который и даёт инверсию профиля
# (см. Таблица 1 курсовой). Значения статьи 2024 года на 2-3 порядка больше
# и с тем же Omega задавят дрейф диффузией - инверсия, скорее всего, исчезнет.
# Если понадобится расчёт в духе статьи 2024 - переопределять D_cleft/D_pm
# нужно вместе с Omega_cleft/Omega_pm, пересчитав число Пекле заново.
D_cleft = 10.0           # коэффициент диффузии внутри синапса, нм²/мкс
D_pm = 0.5               # коэффициент диффузии снаружи синапса (в PM)
D_delta_override = None  # если задать число, D_pm = D_cleft - D_delta_override
if D_delta_override is not None:
    D_pm = D_cleft - float(D_delta_override)

a = 2.00e02              # внутренняя граница синапса, нм
b = 4.00e02              # внешняя граница синапса, нм

D_transition_kind = "sigmoid"   # вид перехода D: "sigmoid" | "poly"
D_transition_steepness = 15.0   # крутизна перехода D
D_drop_start_fraction = 0.05    # доля от |D_pm - D_cleft|, от которой считается начало спада
auto_include_drop_radius = True  # добавлять ли радиус начала спада D в pr_times
inversion_reference_radius = 200.0

# =============================================================================
# [РУЧНОЕ ЗАПОЛНЕНИЕ] Параметры сетки и времени
# =============================================================================
max_r = 1.0e03           # максимальный радиус сетки, нм
dt = 0.5                # шаг по времени, мкс
dr = 0.5                # шаг по пространству, нм
time_divider = 2.0       # делитель для масштабирования времён
base_t_max = 2000.0     # базовое максимальное время, мкс
t_max = base_t_max / time_divider

# =============================================================================
# [РУЧНОЕ ЗАПОЛНЕНИЕ] Параметры инверсии
# =============================================================================
inversion_zone = (a, b)
inversion_min_ratio = 1.05
inversion_peak_floor_fraction = 1e-4
inversion_boundary_guard_nm = 50.0
auto_include_inversion_time = True

# --- Эксайтотоксическая экспозиция (спилловер к внесинаптическим NMDA-R) ---
# I(t) = max(p_вне)/max(p_внутри) - геометрическая характеристика формы
# профиля, но не отвечает на вопрос "сколько медиатора и как долго было
# снаружи". excitotoxicity_extrasynaptic_radius - граница, начиная с
# которой считаем медиатор "внесинаптическим" (по умолчанию - внешний
# край переходной зоны b, т.е. синаптическая щель + её ближайшая
# окрестность). excitotoxicity_concentration_threshold - порог массы,
# выше которого считаем экспозицию патологической; TODO: откалибровать
# по литературе (EC50 активации внесинаптических NMDA-R), сейчас не
# задан (None), метрика считается только как непрерывная доза-время.
excitotoxicity_extrasynaptic_radius = b
excitotoxicity_concentration_threshold = None

# =============================================================================
# [РУЧНОЕ ЗАПОЛНЕНИЕ] Режимы работы
# =============================================================================
compare_omega_cases = False      # True = запускать базовый + утроенный Omega сценарии
theory_scan_enabled = False      # True = сканировать набор D_pm
theory_scan_d_pm_values = (2.0, 5.0, 10.0, 20.0, 40.0, D_cleft)
theory_scan_t_max = t_max
active_omega_case = "q_3q_слева"
outer_boundary_mode = "dirichlet"   # "dirichlet" | "no_flux"
geometry_mode = "spherical"         # "spherical" | "cylindrical"

# --- Нормировка и физическая интерпретация p(r,t) ---
# "probability"   - p(r,t) есть вероятность найти ОДНУ частицу; масса
#                   принудительно нормируется к 1 на каждом шаге и при
#                   каждом импульсе. Это исходный и текущий режим по умолчанию.
# "concentration" - c(r,t) есть концентрация вещества после выброса;
#                   ренормировка отключается, масса меняется только через
#                   поток на границе (outer_boundary_mode), отбор
#                   (k_cleft/k_pm) и источник (импульсы). НЕ смешивать два
#                   режима в одном расчёте: выбор нужно сделать один раз
#                   и использовать его для интерпретации всех результатов.
normalization_mode = "probability"   # "probability" | "concentration"

# --- Отбор (uptake) вещества, k(r) ---
# Слагаемое -k(r)*c в уравнении переноса: захват медиатора транспортёрами
# и т.п. По умолчанию отключено (k=0), чтобы не менять поведение модели,
# использовавшейся ранее. Единицы: 1/мкс. Переход между k_cleft и k_pm
# использует ту же зону [a, b] и тип перехода, что и D(r) (см. ниже).
k_cleft = 0.0
k_pm = 0.0
k_transition_kind = None        # None = использовать D_transition_kind
k_transition_steepness = None   # None = использовать D_transition_steepness

# --- Вид отбора: "linear" (-k*c, выше) | "saturable" (Михаэлис-Ментен) | "none" ---
# "saturable" физически ближе к обоснованию курсовой ("транспортёры
# перегружаются"): скорость отбора -Vmax(r)*c/(Km+c) выходит на плато Vmax
# при c>>Km, то есть в буквальном смысле насыщается/перегружается, в
# отличие от линейного k*c, который никогда не насыщается. Реализовано как
# явная (лаговая) добавка в правую часть, а не перестройка матрицы каждый
# шаг - см. комментарий в solver/fp_solver.py. TODO: Vmax_cleft/Vmax_pm/Km
# ниже - плейсхолдеры (отбор выключен по умолчанию), нужна калибровка по
# литературе (кинетика транспортёров EAAT) прежде чем использовать
# количественно, а не только как демонстрацию механизма.
uptake_kind = "linear"      # "linear" | "saturable" | "none"
Vmax_cleft = 0.0             # максимальная скорость отбора внутри синапса (0 = выключено)
Vmax_pm = 0.0                # максимальная скорость отбора снаружи
Km = 1.0                     # константа полунасыщения (в единицах c/p)

# =============================================================================
# [РУЧНОЕ ЗАПОЛНЕНИЕ] Вывод и экспорт
# =============================================================================
enable_animation = True         # True = показывать анимацию в конце
export_figures = True            # True = сохранять графики
export_data = True               # True = сохранять данные (npz, csv, json)
show_figures_before_save = False
export_plot_tables = True
output_dir = Path("результаты_моделирования")

# =============================================================================
# [РУЧНОЕ ЗАПОЛНЕНИЕ] Параметры информационных мер
# =============================================================================
information_time_window = (0.0, t_max)
information_pairs = {
    "внутри_синапса": (50.0, 150.0),
    "граница": (180.0, 230.0),
    "далеко_от_синапса": (700.0, 900.0),
}
information_hist_bins = 24
rare_event_radius = 950.0
rare_event_quantile = 0.95

# =============================================================================
# [РУЧНОЕ ЗАПОЛНЕНИЕ] Параметры построения графиков
# =============================================================================
pr_times = [
    500.0 / time_divider,
    1500.0 / time_divider,
    5000.0 / time_divider,
    10000.0 / time_divider,
    30000.0 / time_divider,
    50000.0 / time_divider,
    100000.0 / time_divider,
    200000.0 / time_divider,
]
pt_radii = [150.0, 250.0, 350.0, 450.0, 550.0, 650.0]
sigma_r_window = (0.0, max_r)
sigma_t_window = (0.0, t_max)
sigma_kind = "epr"
local_entropy_radius = 200.0
local_entropy_radii = [50.0, local_entropy_radius, 265.0, 850.0]
entropy_probe_radii = [50.0, 150.0, 200.0, 265.0, 300.0, 850.0, 950.0]
entropy_probe_times = [
    500.0 / time_divider,
    1500.0 / time_divider,
    5000.0 / time_divider,
    10000.0 / time_divider,
    30000.0 / time_divider,
    50000.0 / time_divider,
    100000.0 / time_divider,
    200000.0 / time_divider,
]
realtime_pause_sec = 0.01
animation_stride = 50
show_pulse_markers = True
realtime_modeling = True           # True = живой график (медленно!), False = без него
realtime_modeling_stride = animation_stride
realtime_modeling_pause_sec = 0.001

# Режимы графиков p(r,t)
pr_plot_mode = "realtime"       # "realtime" | "overlay"
pr_realtime_window = (0.0, t_max)
pr_realtime_stride = animation_stride
pr_realtime_trail = True
pr_realtime_trail_alpha = 0.2
pr_realtime_trail_max = None
sigma_plot_mode = "lines"       # "lines" | "heatmap"
sigma_times = pr_times
sigma_p_min = 1e-12
sigma_d_min = 1e-12
sigma_clip_percentile = 99.0
sigma_use_log = False

# =============================================================================
# [РУЧНОЕ ЗАПОЛНЕНИЕ] Параметры дрейфа
# Чтобы отключить дрейф: Omega_cleft = 0.0; Omega_pm = 0.0
# =============================================================================
Omega_cleft = 0.05
Omega_pm = 3.0 * Omega_cleft
Omega_pm_tripled = Omega_pm
Omega_transition_kind = "poly"     # "poly" | "sigmoid"
Omega_transition_steepness = 15.0

# =============================================================================
# [РУЧНОЕ ЗАПОЛНЕНИЕ] Параметры импульсов
# Чтобы отключить импульсы: num_pulses = 0 (или pulse_amount = 0)
# =============================================================================
pulse_center_nm = 0.0     # центр гауссова импульса, нм
pulse_sigma_nm = 500.0     # ширина гауссова импульса, нм
pulse_r_window = (0.0, 500.0)   # радиальное окно применения импульса, нм
pulse_amount = 5.0        # доля вероятности, добавляемая каждым импульсом
num_pulses = 100           # количество импульсов (0 = без импульсов)
pulse_max_frequency_hz = 100.0  # максимальная частота импульсов, Гц
pulse_trigger_mode = "schedule"  # "schedule" | "on_decay" | "adaptive"
pulse_trigger_radius = 10.0
pulse_trigger_min_gap = 0.0
pulse_trigger_slope_tol = 0.0
pulse_trigger_require_rise = False

pulse_segment_default_mode = "count"   # "count" | "hz"
pulse_segments = None                   # None = использовать num_pulses равномерно
pulse_time_window = (0.0, 0.5 * t_max)

pulse_response_offsets = [
    0.0 / time_divider,
    25.0 / time_divider,
    50.0 / time_divider,
    200.0 / time_divider,
    500.0 / time_divider,
    1000.0 / time_divider,
    2000.0 / time_divider,
]
pulse_zoom_window_us = 2500.0 / time_divider

rare_synapse_edge_band = (a - 25.0, a + 25.0)
scenario_kl_regions = {
    "внутри_синапса": (0.0, a),
    "край_синапса": rare_synapse_edge_band,
    "далеко_от_синапса": (700.0, 900.0),
    "внешний_хвост": (rare_event_radius, max_r),
}
show_sigma_heatmap = True

# =============================================================================
# Производные параметры (не менять вручную - пересчитываются из a, b)
# =============================================================================
V = np.array([
    [1.0, a, a**2, a**3, a**4, a**5],
    [1.0, b, b**2, b**3, b**4, b**5],
    [0.0, 1.0, 2 * a, 3 * a**2, 4 * a**3, 5 * a**4],
    [0.0, 1.0, 2 * b, 3 * b**2, 4 * b**3, 5 * b**4],
    [0.0, 0.0, 2.0, 6 * a, 12 * a**2, 20 * a**3],
    [0.0, 0.0, 2.0, 6 * b, 12 * b**2, 20 * b**3],
])
_y = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
xi_s = lin.solve(V, _y)

# =============================================================================
# [РУЧНОЕ ЗАПОЛНЕНИЕ] Абсолютные единицы (мкМ) и отбор в физических единицах
# Добавлено к встрече 09.10.2026 - верификация модели на экспериментальных данных.
#
# КАК ВКЛЮЧИТЬ:
#   normalization_mode = "concentration"   (в блоке "Нормировка" выше)
#   -> pulse_amount автоматически станет = molecules_per_release (молекул),
#      и c(r,t) можно перевести в мкМ через utils/units.py:
#      c_мкМ = c * solver_to_uM_factor(extracellular_volume_fraction, geometry_mode)
#
#   uptake_kind = "saturable"              (в блоке "Отбор" выше)
#   + use_physical_uptake_units = True
#   -> Km и Vmax ниже (в мкМ и мкМ/мкс) переводятся в единицы решателя.
#
# В режиме "probability" этот блок НИЧЕГО не меняет - старые расчёты
# воспроизводятся как раньше.
# =============================================================================
molecules_per_release = 3000.0        # молекул глутамата на один выброс.
                                      # Zheng, Scimemi, Rusakov 2008 (Biophys J): 2000-3000,
                                      # модель там брала 3000. Matthews et al. 2022: 7000-8000 на везикулу.
extracellular_volume_fraction = 0.2   # α, доля внеклеточного объёма. Zheng 2008: 0.13-0.20 (CA1).
cleft_height_nm = 20.0                # высота щели; нужна только для geometry_mode = "cylindrical"

# --- Отбор транспортёрами EAAT (Михаэлис-Ментен) в физических единицах ---
use_physical_uptake_units = True
# Km: TODO - НЕ ПРОВЕРЕНО по первоисточнику. В литературе для EAAT встречаются
# значения от ~10 до ~100 мкМ в зависимости от подтипа и препарата. 20 мкМ - рабочая
# оценка; перед защитой взять конкретное число со ссылкой.
Km_uM = 20.0
# Vmax оцениваем как [транспортёры] * скорость оборота цикла (Zheng 2008):
#   эквивалент транспортёров во внеклеточном объёме ~0.2 мМ = 200 мкМ,
#   верхняя скорость цикла k_c ~0.05 мс^-1 = 5e-5 мкс^-1.
transporter_concentration_uM = 200.0
transporter_cycling_rate_per_us = 0.05 / 1000.0
Vmax_pm_uM_per_us = transporter_concentration_uM * transporter_cycling_rate_per_us   # = 0.01 мкМ/мкс
# Внутри щели транспортёров почти нет (Zheng 2008: не больше ~20-30 штук) -> 0.
Vmax_cleft_uM_per_us = 0.0
# ОГРАНИЧЕНИЕ: Михаэлис-Ментен с Vmax по скорости цикла описывает медленный
# стационарный захват. Быстрое связывание глутамата транспортёрами (буферизация,
# субмиллисекундная шкала, Diamond & Jahr 1997) так НЕ описывается - для этого
# нужна кинетическая схема связывания (отдельная задача).

# --- Порог для внесинаптических NMDA-R в мкМ (для verification-скрипта) ---
# Herman, Nahir, Jahr 2011 (PLoS ONE): эквивалент ~0.25 мкМ глутамата уже
# даёт сильный NMDA-Ca2+ сигнал в шипиках. Фон в норме ~0.025 мкМ (Herman & Jahr 2007).
extrasynaptic_nmda_threshold_uM = 0.25
ambient_glutamate_uM = 0.025

if str(normalization_mode).lower() == "concentration":
    from utils.units import uptake_to_solver_units as _uptake_to_solver_units
    pulse_amount = molecules_per_release
    if str(uptake_kind).lower() == "saturable" and use_physical_uptake_units:
        Km, Vmax_cleft, Vmax_pm = _uptake_to_solver_units(
            Km_uM, Vmax_cleft_uM_per_us, Vmax_pm_uM_per_us,
            extracellular_volume_fraction, geometry_mode, cleft_height_nm,
        )
