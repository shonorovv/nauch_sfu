# -*- coding: utf-8 -*-
"""
utils/formatting.py - Утилиты форматирования вывода, сохранения CSV-таблиц и фигур.
"""

import csv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import config


# ---------------------------------------------------------------------------
# Сериализация
# ---------------------------------------------------------------------------

def _to_serializable(value):
    """Рекурсивно конвертирует numpy-типы в стандартные Python-типы для JSON."""
    if isinstance(value, dict):
        return {key: _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(val) for val in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


# ---------------------------------------------------------------------------
# Форматирование значений
# ---------------------------------------------------------------------------

def _format_table_value(value):
    if value is None:
        return ""
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return f"{value:.17g}" if np.isfinite(value) else str(value)
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    return str(value)


def _russian_value(value):
    translations = {
        "center_gaussian": "центральный гауссов профиль",
        "count": "число импульсов",
        "freq": "частота",
        "frequency": "частота",
        "hz": "частота в Гц",
        "on_decay": "по спаду",
        "schedule": "по расписанию",
        "dirichlet": "условие Дирихле",
        "no_flux": "без потока",
        "spherical": "сферическая",
        "cylindrical": "цилиндрическая",
        "unknown": "неизвестно",
        "synapse_edge": "край синапса",
        "outer_tail": "внешний хвост",
    }
    return translations.get(str(value), str(value))


def _russian_key(key):
    translations = {
        "case_name": "имя_сценария",
        "synapse_edge": "край_синапса",
        "outer_tail": "внешний_хвост",
        "case_label": "подпись_сценария",
        "omega_pm": "омега_pm",
        "geometry_mode": "геометрия",
        "D_drop_start_radius": "радиус_начала_спада_D",
        "inv_profile": "инверсия_профиля",
        "nonmonotonic": "немонотонность",
        "peak_inversion": "инверсия_пика",
        "boundary_diagnostics": "диагностика_границы",
        "entropy_checks": "проверки_энтропии",
        "entropy_point_comparison": "сравнение_энтропии_в_точках",
        "info_measures": "информационные_меры",
        "rare_metrics": "редкие_события",
        "rare_region_metrics": "редкие_события_по_областям",
        "pulse_info": "импульсы",
        "center_radius": "радиус_центра",
        "edge_radius": "радиус_края",
        "edge_density_max": "максимум_плотности_на_краю",
        "center_density_max": "максимум_плотности_в_центре",
        "global_density_max": "глобальный_максимум_плотности",
        "max_edge_to_global": "максимум_край_к_общему",
        "max_edge_to_center_valid": "максимум_край_к_центру",
        "time": "время",
        "profile_index": "номер_профиля",
        "peak_radii": "радиусы_пиков",
        "trough_radii": "радиусы_впадин",
        "peak_radius": "радиус_пика",
        "peak_probability": "вероятность_в_пике",
        "entropy_density_at_peak": "энтропия_плотности_в_пике",
        "entropy_density_peak_radius": "радиус_пика_энтропии_плотности",
        "entropy_density_peak_value": "значение_пика_энтропии_плотности",
        "shell_entropy_at_peak": "энтропия_оболочки_в_пике",
        "shell_entropy_peak_radius": "радиус_пика_энтропии_оболочек",
        "shell_entropy_peak_value": "значение_пика_энтропии_оболочек",
        "density_entropy_monotone_theory": "монотонность_энтропии_плотности",
        "radius": "радиус",
        "local_entropy_density": "локальная_энтропия_плотности",
        "shell_probability": "вероятность_оболочки",
        "local_shell_entropy": "локальная_энтропия_оболочки",
        "p_rank": "ранг_p",
        "density_entropy_rank": "ранг_энтропии_плотности",
        "shell_entropy_rank": "ранг_энтропии_оболочки",
        "p_peak_radius": "радиус_пика_p",
        "density_entropy_monotone_region": "область_монотонности_энтропии_плотности",
        "pair": "пара",
        "r_a": "радиус_a",
        "r_b": "радиус_b",
        "kl_ab": "KL_ab",
        "kl_ba": "KL_ba",
        "kl_sym": "симметричная_KL_дивергенция",
        "mutual_information": "взаимная_информация",
        "normalized_mutual_information": "нормированная_взаимная_информация",
        "correlation": "корреляция",
        "tail_radius": "радиус_хвоста",
        "tail_mass_series": "ряд_массы_хвоста",
        "edge_series": "ряд_края",
        "tail_mass_mean": "средняя_масса_хвоста",
        "tail_mass_max": "максимальная_масса_хвоста",
        "edge_mean": "среднее_края",
        "edge_max": "максимум_края",
        "region": "область",
        "mass_series": "ряд_массы",
        "mass_mean": "средняя_масса",
        "mass_max": "максимальная_масса",
        "mass_min": "минимальная_масса",
        "tail_threshold": "порог_хвоста",
        "edge_threshold": "порог_края",
        "reference_tail_rate": "частота_хвоста_базовая",
        "candidate_tail_rate": "частота_хвоста_сравниваемая",
        "reference_edge_rate": "частота_края_базовая",
        "candidate_edge_rate": "частота_края_сравниваемая",
        "reference_rate": "частота_базовая",
        "candidate_rate": "частота_сравниваемая",
        "reference_mean": "среднее_базовое",
        "candidate_mean": "среднее_сравниваемое",
        "reference_max": "максимум_базовый",
        "candidate_max": "максимум_сравниваемый",
        "reference_case": "базовый_сценарий",
        "candidate_case": "сравниваемый_сценарий",
        "kl_reference_candidate": "KL_базовый_к_сравниваемому",
        "kl_candidate_reference": "KL_сравниваемый_к_базовому",
        "reference_mean_mass": "средняя_масса_базовая",
        "candidate_mean_mass": "средняя_масса_сравниваемая",
        "segments": "сегменты",
        "count": "число",
        "pulse_amount": "амплитуда_импульса",
        "source_model": "модель_источника",
        "center_nm": "центр_нм",
        "sigma_nm": "сигма_нм",
        "r_window": "окно_r",
        "time_window": "окно_времени",
        "frequency_window": "окно_частоты",
        "event_rate_hz": "интенсивность_Гц",
        "repetition_rate_hz": "частота_повторения_Гц",
        "min_gap_us": "минимальный_интервал_мкс",
        "mean_gap_us": "средний_интервал_мкс",
        "max_repetition_frequency_hz": "максимальная_частота_повторения_Гц",
        "max_frequency_hz": "максимальная_частота_Гц",
        "segment_default_mode": "режим_сегментов_по_умолчанию",
        "trigger_mode": "режим_запуска",
        "trigger_radius": "радиус_запуска",
        "trigger_min_gap": "минимальный_интервал_запуска",
        "trigger_slope_tol": "допуск_наклона_запуска",
        "trigger_require_rise": "требовать_рост_перед_запуском",
        "outer_boundary_mode": "условие_внешней_границы",
        "uptake_kind": "вид_отбора",
        "Vmax_cleft": "Vmax_внутри",
        "Vmax_pm": "Vmax_снаружи",
        "Km": "константа_полунасыщения",
        "solver_convergence": "сходимость_решателя",
        "total_steps": "всего_шагов",
        "check_stride": "шаг_проверки_невязки",
        "checked_steps": "проверенные_шаги",
        "converged_steps": "сошедшиеся_шаги",
        "failed_steps": "несошедшиеся_шаги",
        "failed_times": "времена_несходимости",
        "residual_tol": "допуск_невязки",
        "extrasynaptic_radius": "внесинаптический_радиус",
        "exposure_total": "суммарная_экспозиция",
        "peak_mass_extra": "пик_внесинаптической_массы",
        "peak_time": "время_пика",
        "concentration_threshold": "порог_концентрации",
        "time_above_threshold": "время_выше_порога",
        "first_time_above_threshold": "первое_время_выше_порога",
    }
    return translations.get(str(key), str(key))


def _translate_saved_value(value):
    """Рекурсивно переводит ключи и строковые значения словаря на русский."""
    if isinstance(value, dict):
        return {_russian_key(key): _translate_saved_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_translate_saved_value(val) for val in value]
    if isinstance(value, tuple):
        return tuple(_translate_saved_value(val) for val in value)
    if isinstance(value, str):
        return _russian_value(value)
    return value


# ---------------------------------------------------------------------------
# Построение CSV-строк для таблиц графиков
# ---------------------------------------------------------------------------

def _append_curve_rows(rows, panel, x_name, x_values, y_name, y_values, series):
    for x_val, y_val in zip(np.asarray(x_values).ravel(), np.asarray(y_values).ravel()):
        rows.append({
            "panel": panel,
            "series": series,
            "x_name": x_name,
            "x_value": _format_table_value(x_val),
            "y_name": y_name,
            "y_value": _format_table_value(y_val),
        })


def _append_event_rows(rows, panel, t_values, series="импульс"):
    for t_val in np.asarray(t_values, dtype=float).ravel():
        rows.append({
            "panel": panel,
            "series": series,
            "x_name": "t, мкс",
            "x_value": _format_table_value(t_val),
            "y_name": "событие",
            "y_value": "1",
        })


def _append_bar_rows(rows, panel, labels, values, y_name, series="столбец"):
    for label, value in zip(labels, values):
        rows.append({
            "panel": panel,
            "series": series,
            "x_name": "категория",
            "x_value": str(label),
            "y_name": y_name,
            "y_value": _format_table_value(value),
        })


# ---------------------------------------------------------------------------
# Вспомогательные функции для графиков
# ---------------------------------------------------------------------------

def _set_common_time_axis(axes, t_values):
    for ax in np.ravel(axes):
        ax.set_xlim(float(t_values[0]), float(t_values[-1]))


def _first_response_delay(t_values, series, pulse_times, threshold_fraction=0.05):
    """Время первого превышения порога после первого импульса."""
    pulse_times = np.asarray(pulse_times, dtype=float)
    if pulse_times.size == 0:
        return None
    t0 = float(pulse_times[0])
    start_idx = int(np.searchsorted(t_values, t0, side="left"))
    start_idx = int(np.clip(start_idx, 0, len(t_values) - 1))
    baseline = float(series[start_idx - 1]) if start_idx > 0 else 0.0
    tail = np.asarray(series[start_idx:], dtype=float)
    if tail.size == 0:
        return None
    peak = float(np.max(tail))
    if peak <= baseline + config.EPS:
        return None
    threshold = baseline + float(threshold_fraction) * (peak - baseline)
    response = np.where(tail >= threshold)[0]
    if response.size == 0:
        return None
    response_idx = start_idx + int(response[0])
    return float(t_values[response_idx] - t0)


# ---------------------------------------------------------------------------
# Сохранение таблиц и фигур
# ---------------------------------------------------------------------------

def _save_plot_table(path, rows):
    if not config.export_plot_tables or not rows:
        return
    csv_path = Path(path).with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["панель", "серия", "имя_x", "значение_x", "имя_y", "значение_y"]
    translated_rows = [
        {
            "панель": row.get("panel", ""),
            "серия": row.get("series", ""),
            "имя_x": row.get("x_name", ""),
            "значение_x": row.get("x_value", ""),
            "имя_y": row.get("y_name", ""),
            "значение_y": row.get("y_value", ""),
        }
        for row in rows
    ]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(translated_rows)


def _can_show_figures_before_save():
    backend = str(matplotlib.get_backend()).lower()
    return bool(config.show_figures_before_save) and "agg" not in backend


def _figure_is_open(fig):
    return plt.fignum_exists(fig.number)


def _show_figure(fig):
    fig.canvas.draw()
    plt.show(block=False)
    plt.pause(0.001)


def _wait_for_close(fig, poll_sec=0.05):
    while _figure_is_open(fig):
        plt.pause(poll_sec)


def _save_figure(fig, path, table_rows=None):
    """Сохраняет matplotlib-фигуру в файл (с опциональным показом перед сохранением)."""
    fig.tight_layout()
    if _can_show_figures_before_save():
        print(f"Открываю график: {Path(path).name}. Закройте окно, чтобы сохранить файл.")
        _show_figure(fig)
        _wait_for_close(fig)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    _save_plot_table(path, table_rows or [])
    plt.close(fig)
