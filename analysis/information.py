# -*- coding: utf-8 -*-
"""
analysis/information.py - Информационные меры: KL-дивергенция, взаимная информация,
вероятности оболочек, редкие события.
"""

import numpy as np
import config
from utils.helpers import _sanitize_window


# ---------------------------------------------------------------------------
# Внутренние вспомогательные функции
# ---------------------------------------------------------------------------

def _normalize_distribution(series):
    arr = np.maximum(np.asarray(series, dtype=float), 0.0)
    total = float(np.sum(arr))
    if total <= 0.0:
        if arr.size == 0:
            return arr
        return np.full(arr.shape, 1.0 / arr.size, dtype=float)
    return arr / total


def _histogram_bins(series, bins):
    series = np.asarray(series, dtype=float)
    if series.size == 0:
        return np.linspace(0.0, 1.0, bins + 1)
    lo = float(np.min(series))
    hi = float(np.max(series))
    if hi <= lo + config.EPS:
        hi = lo + 1.0
    edges = np.quantile(series, np.linspace(0.0, 1.0, bins + 1))
    edges[0] = lo
    edges[-1] = hi
    edges = np.unique(edges)
    if edges.size < 3:
        edges = np.linspace(lo, hi, max(3, bins + 1))
    return edges


# ---------------------------------------------------------------------------
# Публичные функции
# ---------------------------------------------------------------------------

def kl_divergence(p_dist, q_dist):
    """KL(P || Q) = sum p * log(p / q). Распределения нормируются автоматически."""
    p_safe = np.maximum(_normalize_distribution(p_dist), config.EPS)
    q_safe = np.maximum(_normalize_distribution(q_dist), config.EPS)
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


def mutual_information(series_a, series_b, bins=24):
    """
    Оценка взаимной информации I(A;B) через совместную гистограмму.
    Возвращает (mi, normalized_mi).
    """
    a = np.asarray(series_a, dtype=float)
    b = np.asarray(series_b, dtype=float)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0, 0.0
    bins_a = _histogram_bins(a, bins)
    bins_b = _histogram_bins(b, bins)
    hist_2d, _, _ = np.histogram2d(a, b, bins=[bins_a, bins_b])
    total = float(np.sum(hist_2d))
    if total <= 0.0:
        return 0.0, 0.0
    p_xy = hist_2d / total
    p_x = np.sum(p_xy, axis=1, keepdims=True)
    p_y = np.sum(p_xy, axis=0, keepdims=True)
    mask = p_xy > 0.0
    mi = float(np.sum(
        p_xy[mask] * np.log(p_xy[mask] / np.maximum((p_x @ p_y)[mask], config.EPS))
    ))
    h_x = float(-np.sum(p_x[p_x > 0.0] * np.log(p_x[p_x > 0.0])))
    h_y = float(-np.sum(p_y[p_y > 0.0] * np.log(p_y[p_y > 0.0])))
    norm = max(min(h_x, h_y), config.EPS)
    return mi, float(mi / norm)


def compute_information_measures(
    t_values,
    r_values,
    shell_probabilities,
    pairs,
    time_window=None,
    bins=24,
):
    """
    Вычисляет KL-дивергенцию, взаимную информацию и корреляцию
    между парами точек по временным рядам вероятностей оболочек.

    pairs - словарь {name: (r_a, r_b)}.
    """
    t_values = np.asarray(t_values, dtype=float)
    if time_window is None:
        mask_t = np.ones(t_values.shape, dtype=bool)
    else:
        t_start, t_end = _sanitize_window(time_window, float(t_values[0]), float(t_values[-1]))
        mask_t = (t_values >= t_start - config.TIME_TOL) & (t_values <= t_end + config.TIME_TOL)
    results = []
    for pair_name, (r_a, r_b) in pairs.items():
        idx_a = int(np.argmin(np.abs(r_values - float(r_a))))
        idx_b = int(np.argmin(np.abs(r_values - float(r_b))))
        series_a = np.asarray(shell_probabilities[idx_a, mask_t], dtype=float)
        series_b = np.asarray(shell_probabilities[idx_b, mask_t], dtype=float)
        kl_ab = kl_divergence(series_a, series_b)
        kl_ba = kl_divergence(series_b, series_a)
        mi, norm_mi = mutual_information(series_a, series_b, bins=bins)
        corr = float(np.corrcoef(series_a, series_b)[0, 1]) if series_a.size > 1 else 0.0
        if not np.isfinite(corr):
            corr = 0.0
        results.append({
            "pair": pair_name,
            "r_a": float(r_values[idx_a]),
            "r_b": float(r_values[idx_b]),
            "kl_ab": kl_ab,
            "kl_ba": kl_ba,
            "kl_sym": 0.5 * (kl_ab + kl_ba),
            "mutual_information": mi,
            "normalized_mutual_information": norm_mi,
            "correlation": corr,
        })
    return results


def compute_rare_event_metrics(shell_probabilities, r_values, tail_radius):
    """Статистика «редких событий» в хвосте распределения (r >= tail_radius)."""
    r_values = np.asarray(r_values, dtype=float)
    tail_mask = r_values >= float(tail_radius)
    if not np.any(tail_mask):
        tail_mask = r_values >= float(np.max(r_values))
    tail_mass = np.sum(shell_probabilities[tail_mask, :], axis=0)
    edge_idx = int(np.argmax(r_values))
    edge_series = shell_probabilities[edge_idx, :]
    return {
        "tail_radius": float(r_values[np.argmax(tail_mask)]),
        "tail_mass_series": tail_mass,
        "edge_radius": float(r_values[edge_idx]),
        "edge_series": edge_series,
        "tail_mass_mean": float(np.mean(tail_mass)),
        "tail_mass_max": float(np.max(tail_mass)),
        "edge_mean": float(np.mean(edge_series)),
        "edge_max": float(np.max(edge_series)),
    }


def compute_region_event_metrics(shell_probabilities, r_values, region):
    """Суммарная вероятность в заданной области [r_min, r_max] по времени."""
    r_values = np.asarray(r_values, dtype=float)
    r_min, r_max = _sanitize_window(region, float(r_values[0]), float(r_values[-1]))
    mask = (r_values >= r_min - config.TIME_TOL) & (r_values <= r_max + config.TIME_TOL)
    if not np.any(mask):
        idx = int(np.argmin(np.abs(r_values - 0.5 * (r_min + r_max))))
        mask = np.zeros_like(r_values, dtype=bool)
        mask[idx] = True
    series = np.sum(shell_probabilities[mask, :], axis=0)
    return {
        "region": (float(r_values[mask][0]), float(r_values[mask][-1])),
        "mass_series": series,
        "mass_mean": float(np.mean(series)),
        "mass_max": float(np.max(series)),
        "mass_min": float(np.min(series)),
    }


def compare_region_events(reference_metrics, candidate_metrics, quantile=0.95):
    """
    Сравнивает частоту попаданий в пороговую зону между
    базовым и сравниваемым сценарием по нескольким областям.
    """
    comparison = {}
    for region_name, ref_metric in reference_metrics.items():
        cand_metric = candidate_metrics.get(region_name)
        if cand_metric is None:
            continue
        threshold = float(np.quantile(ref_metric["mass_series"], quantile))
        comparison[region_name] = {
            "threshold": threshold,
            "reference_rate": float(np.mean(ref_metric["mass_series"] >= threshold)),
            "candidate_rate": float(np.mean(cand_metric["mass_series"] >= threshold)),
            "reference_mean": float(ref_metric["mass_mean"]),
            "candidate_mean": float(cand_metric["mass_mean"]),
            "reference_max": float(ref_metric["mass_max"]),
            "candidate_max": float(cand_metric["mass_max"]),
            "region": ref_metric["region"],
        }
    return comparison


def compare_rare_events(reference_metrics, candidate_metrics, quantile=0.95):
    """
    Сравнивает хвостовые события и события на краю сетки
    между базовым и сравниваемым сценарием.
    """
    tail_threshold = float(np.quantile(reference_metrics["tail_mass_series"], quantile))
    edge_threshold = float(np.quantile(reference_metrics["edge_series"], quantile))
    return {
        "tail_threshold": tail_threshold,
        "edge_threshold": edge_threshold,
        "reference_tail_rate": float(np.mean(reference_metrics["tail_mass_series"] >= tail_threshold)),
        "candidate_tail_rate": float(np.mean(candidate_metrics["tail_mass_series"] >= tail_threshold)),
        "reference_edge_rate": float(np.mean(reference_metrics["edge_series"] >= edge_threshold)),
        "candidate_edge_rate": float(np.mean(candidate_metrics["edge_series"] >= edge_threshold)),
    }
