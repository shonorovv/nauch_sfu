# -*- coding: utf-8 -*-
"""
export/data.py - Сохранение числовых массивов (npz) и JSON-сводки для одного сценария.
"""

import json
import numpy as np
import config
from utils.formatting import _to_serializable, _russian_value, _translate_saved_value


def export_scenario_data(scenario, scenario_dir):
    """
    Сохраняет:
    - массивы.npz   - все временные ряды (p, энтропия, sigma, ...)
    - сводка.json   - сериализуемые метаданные сценария

    Запись npz управляется флагом config.export_data.
    JSON-сводка всегда сохраняется.
    """
    from solver.geometry import normalize_geometry_mode  # ленивый импорт

    scenario_dir.mkdir(parents=True, exist_ok=True)

    if config.export_data:
        np.savez_compressed(
            scenario_dir / "массивы.npz",
            **{
                "радиусы_нм": scenario["r_values"],
                "время_мкс": scenario["t_values"],
                "история_p": scenario["p_history"],
                "локальная_энтропия_плотности": scenario["local_entropy_density"],
                "локальная_энтропия_оболочек": scenario["local_shell_entropy"],
                "история_сигма": scenario["sigma_history"],
                "вероятности_оболочек": scenario["shell_probabilities"],
            },
        )

    summary = {
        "имя_сценария": scenario["case_name"],
        "подпись_сценария": scenario["case_label"],
        "омега_pm": scenario["omega_pm"],
        "геометрия": _russian_value(scenario.get("geometry_mode", normalize_geometry_mode())),
        "радиус_начала_спада_D": scenario.get("D_drop_start_radius"),
        "инверсия_профиля": _translate_saved_value(scenario["inv_profile"]),
        "немонотонность": _translate_saved_value(scenario["nonmonotonic"]),
        "инверсия_пика": _translate_saved_value(scenario["peak_inversion"]),
        "диагностика_границы": _translate_saved_value(scenario.get("boundary_diagnostics", {})),
        "проверки_энтропии": _translate_saved_value(scenario["entropy_checks"]),
        "сравнение_энтропии_в_точках": _translate_saved_value(scenario["entropy_point_comparison"]),
        "информационные_меры": _translate_saved_value(scenario["info_measures"]),
        "редкие_события": _translate_saved_value({
            key: value
            for key, value in scenario["rare_metrics"].items()
            if not isinstance(value, np.ndarray)
        }),
        "редкие_события_по_областям": _translate_saved_value({
            region_name: {
                key: value
                for key, value in metrics.items()
                if not isinstance(value, np.ndarray)
            }
            for region_name, metrics in scenario["rare_region_metrics"].items()
        }),
        "импульсы": _translate_saved_value({
            key: _russian_value(value) if isinstance(value, str) else value
            for key, value in scenario["pulse_info"].items()
            if key not in ("indices", "pulse_times")
        }),
    }
    with open(scenario_dir / "сводка.json", "w", encoding="utf-8") as fh:
        json.dump(_to_serializable(summary), fh, ensure_ascii=False, indent=2)
