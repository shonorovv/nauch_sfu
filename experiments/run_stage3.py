# -*- coding: utf-8 -*-
"""
experiments/run_stage3.py - этап 3 ТЗ от 13.10.2026: контрольные режимы монотонности и устойчивость инверсии.

Каждый контроль из docs/monotonicity_note.md (п. 5) считается трижды: базовая сетка, сетка dr и dt вдвое
мельче, граница R в 1,5 раза дальше. Для каждого прогона детектор инверсии ЛОКАЛЬНОЙ КОНЦЕНТРАЦИИ
(analysis/inversion_concentration.py) запускается с двумя абсолютными порогами: 1e-6 и 1e-3 мкМ.
Критерий устойчивости (п. 5 этапа 3): знак (найдена / не найдена) одинаков во всех прогонах,
а у найденных ΔC отличается от базовой не более чем на 5%.

Результат: experiments_outputs/stage3/inversion_controls.xlsx, inversion_controls.json,
inversion_controls.png, manifest.json. CSV не используется.

Запуск из папки diplom:  python -m experiments.run_stage3
"""
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from experiments import configs as C
from experiments.run_stage02 import run_case, save_xlsx, git_sha, OUT as _STAGE02_OUT  # noqa: F401  (импорт задаёт CPU-режим)
from analysis.inversion_concentration import concentration_inversion
import scipy
import matplotlib.pyplot as plt

OUT = Path("experiments_outputs/stage3")
THRESHOLDS_UM = (1e-6, 1e-3)
DELTA_TOL = 0.05


def _variant(name, base, **kw):
    p = C.get(base)
    p.update(kw)
    p["name"] = name
    return p


def controls():
    """Контроли 1-5 раздела 3 ТЗ; R и T подобраны так, чтобы граница не влияла на c у синапса."""
    short = dict(R_nm=3000.0, t_max_us=2000.0)
    return [
        ("1", "однородная диффузия, без захвата и дрейфа", "нет",
         _variant("c1", "baseline_physical", D_cleft=320.0, D_pm=320.0, **short)),
        ("2a", "D_cleft = 10 < D_pm, геометрия Barbour", "нет",
         _variant("c2a", "baseline_physical", D_cleft=10.0, D_pm=320.0, **short)),
        ("2b", "D_cleft = 760 > D_pm, геометрия Barbour", "нет",
         _variant("c2b", "baseline_physical", D_cleft=760.0, D_pm=320.0, **short)),
        ("3", "насыщаемый захват, не убывающий по r, без дрейфа", "нет",
         _variant("c3", "with_uptake", D_cleft=10.0, Vmax_pm=0.01, **short)),
        ("4", "курсовая: дрейф наружу Ω = 0,05/0,15 (сфера)", "возможна",
         C.get("legacy_coursework")),
        ("4b", "то же без дрейфа", "нет",
         _variant("c4b", "legacy_coursework", Omega_cleft=0.0, Omega_pm=0.0)),
        ("5", "дрейф внутрь из единой подгонки (Ω = -0,084/-0,25)", "нет",
         _variant("c5", "baseline_physical", D_cleft=16.7, Omega_cleft=-0.084, Omega_pm=-0.25, **short)),
    ]


def detect(res, thr):
    return concentration_inversion(res["c"], res["r"], res["t"], abs_threshold_uM=thr)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    rows, detail, curves = [], {}, {}
    for cid, label, expected, p in controls():
        runs = {
            "base": run_case(p),
            "fine": run_case(p, dr=p["dr_nm"] / 2.0, dt=p["dt_us"] / 2.0),
            "R x1.5": run_case(p, R=p["R_nm"] * 1.5),
        }
        curves[cid] = (label, runs["base"])
        detail[cid] = {"label": label, "expected": expected, "params": p, "runs": {}}
        for thr in THRESHOLDS_UM:
            res = {k: detect(v, thr) for k, v in runs.items()}
            base = res["base"]
            same_sign = all(r["found"] == base["found"] for r in res.values())
            if base["found"] and same_sign:
                d0 = base["delta_c_uM"]
                dev = max(abs(r["delta_c_uM"] - d0) / d0 for r in res.values())
            else:
                dev = 0.0
            stable = same_sign and dev <= DELTA_TOL
            ok = base["found"] == (expected == "возможна")
            for k, r in res.items():
                rows.append([cid, label, expected, thr, k, r["found"], r["t_first_us"], r["r_left_nm"], r["r_right_nm"],
                             r["delta_c_uM"], r["ratio_c"], r["duration_us"],
                             r["max_positive_gradient_uM_per_nm"], r["peak_shift_nm"]])
            detail[cid]["runs"][str(thr)] = {k: r for k, r in res.items()} | {
                "stable": bool(stable), "max_delta_dev": float(dev), "matches_expectation": bool(ok)}
            print(f"[{cid}] порог {thr:g} мкМ: найдена={base['found']} (ожидалось: {expected}), "
                  f"устойчиво={stable}, макс. откл. ΔC={dev:.2%}, {'PASS' if ok and stable else 'FAIL'}")

    header = ["контроль", "описание", "ожидание", "порог_мкМ", "прогон", "найдена", "t_first_мкс", "r_left_нм",
              "r_right_нм", "ΔC_мкМ", "c_right/c_left", "длительность_мкс", "макс_градиент_мкМ/нм", "сдвиг_пика_нм"]
    save_xlsx(OUT / "inversion_controls.xlsx", {"inversion_controls": (header, rows)})
    (OUT / "inversion_controls.json").write_text(json.dumps(detail, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5)); axes = axes.ravel()
    for ax, (cid, (label, res)) in zip(axes, curves.items()):
        ks = np.unique(np.linspace(0, res["t"].size - 1, 6).astype(int))
        for k in ks:
            ax.semilogy(res["r"], np.maximum(res["c"][:, k], 1e-12), label=f"t = {res['t'][k]:.0f} мкс")
        ax.set_xlim(0, min(1200.0, res["r"][-1])); ax.set_ylim(1e-6, None)
        ax.set_title(f"контроль {cid}: {label}", fontsize=8)
        ax.set_xlabel("r, нм"); ax.set_ylabel("c, мкМ"); ax.legend(fontsize=6)
    for ax in axes[len(curves):]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "inversion_controls.png", dpi=130); plt.close(fig)

    all_ok = all(v["stable"] and v["matches_expectation"] for d in detail.values() for v in
                 (d["runs"][str(t)] for t in THRESHOLDS_UM))
    (OUT / "manifest.json").write_text(json.dumps({
        "git_sha": git_sha(), "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(), "device": "CPU", "all_checks_pass": bool(all_ok),
        "command": "python -m experiments.run_stage3", "time": time.strftime("%Y-%m-%d %H:%M:%S")},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("ИТОГ:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
