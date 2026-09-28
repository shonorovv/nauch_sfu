# -*- coding: utf-8 -*-
"""
tests/test_validation.py - Минимальный блок валидации решателя Фоккера-Планка.

Проверяет, что "инверсия профиля" и прочие эффекты - свойство модели,
а не артефакт сетки, источника или границы:

- test_pure_diffusion_second_moment_matches_theory
    Чистая диффузия (Omega=0, k=0, один импульс как начальное условие,
    без дальнейших импульсов), сравнение с полуаналитическим результатом:
    для отражающей (no-flux) границы и постоянного D тождество
    d/dt E[r^2] = 2*d*D (d - размерность геометрии) точное (формула
    Дынкина для отражённого броуновского движения), не приближение.

- test_mass_conservation_without_forced_renormalization
    Постоянный D, нулевой Omega, no-flux граница, normalization_mode=
    "concentration" (без принудительной ренормировки, см. config.py) -
    масса должна сохраняться сама по себе, а не за счёт деления на total.

- test_non_negativity
    p(r,t) >= 0 во всех точках сетки и всех сохранённых кадрах.

- test_grid_convergence
    Сходимость по сетке: измельчение (dr, dt) -> (dr/2, dt/2) -> (dr/4, dt/4)
    должно приближать решение к общему пределу (разница между соседними
    уровнями измельчения убывает).

Дополнительно (быстрые unit-тесты для конкретных замечаний руководителя):

- test_build_omega_case_table_include_comparison
- test_find_mass_inversion_time_uses_region_mass
"""

import os
import sys
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config  # noqa: E402
config.use_gpu = False  # для тестов используем CPU/numpy бэкенд
config.realtime_modeling = False

from solver.fp_solver import solve_fp_equation  # noqa: E402
from solver.geometry import radial_cell_measures  # noqa: E402


def _solve(max_r, dt, dr, t_max, D, geometry_mode="spherical", **overrides):
    """Запускает решатель с "чистой" диффузией (без дрейфа и отбора по умолчанию)."""
    config.geometry_mode = geometry_mode
    params = dict(
        max_r=max_r,
        dt=dt,
        dr=dr,
        t_max=t_max,
        D_cleft=D,
        D_pm=D,
        xi_s=config.xi_s,
        a=config.a,
        b=config.b,
        Omega_cleft=0.0,
        Omega_pm=0.0,
        pulse_amount=1.0,
        pulse_r_window=(0.0, 5.0 * max(dr, 1.0)),
        num_pulses=1,
        pulse_segments=None,
        pulse_time_window=(0.0, 0.0),
        pulse_segment_default_mode="count",
        pulse_trigger_mode="schedule",
        pulse_trigger_radius=10.0,
        pulse_trigger_min_gap=0.0,
        pulse_trigger_slope_tol=0.0,
        pulse_trigger_require_rise=False,
        D_transition_kind="sigmoid",
        D_transition_steepness=15.0,
        Omega_transition_kind="poly",
        Omega_transition_steepness=15.0,
        outer_boundary_mode="no_flux",
        k_cleft=0.0,
        k_pm=0.0,
        normalization_mode="concentration",
    )
    params.update(overrides)
    return solve_fp_equation(**params)


def _mass_and_second_moment(p_history, r_values, dr, geometry_mode):
    dV = radial_cell_measures(r_values, dr, geometry_mode)
    mass = np.sum(p_history * dV[:, None], axis=0)
    r2 = np.sum(p_history * (r_values ** 2)[:, None] * dV[:, None], axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        second_moment = r2 / np.maximum(mass, 1e-300)
    return mass, second_moment


class TestPureDiffusion(unittest.TestCase):
    """Тест 1 из минимального набора: чистая диффузия, известное поведение ширины."""

    def test_pure_diffusion_second_moment_matches_theory_spherical(self):
        D = 40.0
        dr = 1.0
        p_history, r_values, t_saved, _, _, pulse_info = _solve(
            max_r=400.0, dt=0.05, dr=dr, t_max=200.0, D=D, geometry_mode="spherical",
        )
        _, second_moment = _mass_and_second_moment(p_history, r_values, dr, "spherical")

        # Пропускаем самый первый кадр (ещё не рассосавшийся резкий импульс).
        t_fit = t_saved[1:]
        m2_fit = second_moment[1:]
        slope, intercept = np.polyfit(t_fit, m2_fit, 1)

        expected_slope = 2.0 * 3.0 * D  # d/dt E[r^2] = 2*d*D, d=3 (сферическая геометрия)
        rel_err = abs(slope - expected_slope) / expected_slope
        self.assertLess(
            rel_err, 0.05,
            f"Наклон E[r^2](t) = {slope:.3f} далёк от теории 2*d*D = {expected_slope:.3f} "
            f"(отн. ошибка {rel_err:.3%})",
        )

    def test_pure_diffusion_second_moment_matches_theory_cylindrical(self):
        D = 30.0
        dr = 1.0
        p_history, r_values, t_saved, _, _, pulse_info = _solve(
            max_r=400.0, dt=0.05, dr=dr, t_max=200.0, D=D, geometry_mode="cylindrical",
        )
        _, second_moment = _mass_and_second_moment(p_history, r_values, dr, "cylindrical")

        t_fit = t_saved[1:]
        m2_fit = second_moment[1:]
        slope, intercept = np.polyfit(t_fit, m2_fit, 1)

        expected_slope = 2.0 * 2.0 * D  # d=2 (цилиндрическая геометрия)
        rel_err = abs(slope - expected_slope) / expected_slope
        self.assertLess(
            rel_err, 0.05,
            f"Наклон E[r^2](t) = {slope:.3f} далёк от теории 2*d*D = {expected_slope:.3f} "
            f"(отн. ошибка {rel_err:.3%})",
        )


class TestMassConservation(unittest.TestCase):
    """Тест 2 из минимального набора: сохранение массы без принудительной ренормировки."""

    def test_mass_conservation_without_forced_renormalization(self):
        D = 60.0
        dr = 1.0
        p_history, r_values, t_saved, _, _, pulse_info = _solve(
            max_r=300.0, dt=0.1, dr=dr, t_max=150.0, D=D, geometry_mode="spherical",
            normalization_mode="concentration",
            outer_boundary_mode="no_flux",
        )
        self.assertEqual(pulse_info["normalization_mode"], "concentration")

        mass, _ = _mass_and_second_moment(p_history, r_values, dr, "spherical")
        mass0 = mass[0]
        rel_dev = np.max(np.abs(mass - mass0)) / mass0
        self.assertLess(
            rel_dev, 1e-6,
            f"Масса не сохраняется без ренормировки: относительное отклонение {rel_dev:.3e}",
        )

    def test_uptake_removes_mass_at_expected_rate(self):
        """Отбор k(r) должен убирать массу; без него (k=0) масса сохраняется (см. выше)."""
        D = 60.0
        dr = 1.0
        dt = 0.1
        k = 0.2
        t_max = 50.0
        p_history, r_values, t_saved, _, _, pulse_info = _solve(
            max_r=300.0, dt=dt, dr=dr, t_max=t_max, D=D, geometry_mode="spherical",
            normalization_mode="concentration",
            outer_boundary_mode="no_flux",
            k_cleft=k, k_pm=k,
        )
        mass, _ = _mass_and_second_moment(p_history, r_values, dr, "spherical")
        # Неявная (backward Euler) дискретизация стока даёт mass(t) = mass0 * (1+k*dt)^(-t/dt).
        expected_final = mass[0] * (1.0 + k * dt) ** (-t_saved[-1] / dt)
        rel_err = abs(mass[-1] - expected_final) / expected_final
        self.assertLess(rel_err, 1e-6)
        self.assertLess(mass[-1], mass[0])


class TestSaturableUptake(unittest.TestCase):
    """
    Регрессия для насыщаемого отбора (Михаэлис-Ментен), см.
    solver/fp_solver.py и config.py (uptake_kind="saturable"). Реализован
    как явная (лаговая) добавка в правую часть, поэтому для первого шага
    убыль массы должна ТОЧНО совпадать с dt * sum(Vmax*p0/(Km+p0)*dV) -
    диффузия/дрейф с no-flux границей сами по себе массу сохраняют, так что
    вся убыль массы объясняется реакцией.
    """

    def test_saturable_uptake_first_step_mass_loss_matches_explicit_estimate(self):
        D = 60.0
        dr = 1.0
        dt = 0.05
        Vmax = 0.01
        Km = 0.5
        p_history, r_values, t_saved, _, _, pulse_info = _solve(
            max_r=300.0, dt=dt, dr=dr, t_max=dt, D=D, geometry_mode="spherical",
            normalization_mode="concentration",
            outer_boundary_mode="no_flux",
            uptake_kind="saturable",
            Vmax_cleft=Vmax, Vmax_pm=Vmax,
            Km=Km,
        )
        self.assertEqual(pulse_info["uptake_kind"], "saturable")
        self.assertEqual(t_saved.size, 2)

        dV = radial_cell_measures(r_values, dr, "spherical")
        mass0 = float(np.sum(p_history[:, 0] * dV))
        mass1 = float(np.sum(p_history[:, 1] * dV))
        reaction0 = Vmax * p_history[:, 0] / (Km + p_history[:, 0])
        expected_mass1 = mass0 - dt * float(np.sum(reaction0 * dV))

        rel_err = abs(mass1 - expected_mass1) / mass0
        self.assertLess(rel_err, 1e-8)
        self.assertLess(mass1, mass0)

    def test_uptake_kind_rejects_unknown_value(self):
        with self.assertRaises(ValueError):
            _solve(
                max_r=100.0, dt=0.1, dr=1.0, t_max=1.0, D=10.0,
                uptake_kind="not_a_real_mode",
            )


class TestNonNegativity(unittest.TestCase):
    """Часть минимального блока валидации: p(r,t) не должно уходить в отрицательные значения."""

    def test_non_negativity(self):
        p_history, r_values, t_saved, _, _, _ = _solve(
            max_r=300.0, dt=0.1, dr=1.0, t_max=100.0, D=50.0, geometry_mode="spherical",
        )
        self.assertGreaterEqual(float(np.min(p_history)), 0.0)


class TestGridConvergence(unittest.TestCase):
    """Тест 3 из минимального набора: сеточная сходимость dr, dr/2 и dt, dt/2."""

    def test_grid_convergence_second_moment(self):
        D = 50.0
        max_r = 200.0
        t_max = 60.0

        def diagnostic(dr, dt):
            p_history, r_values, t_saved, _, _, _ = _solve(
                max_r=max_r, dt=dt, dr=dr, t_max=t_max, D=D, geometry_mode="spherical",
            )
            _, second_moment = _mass_and_second_moment(p_history, r_values, dr, "spherical")
            return float(second_moment[-1])

        coarse = diagnostic(dr=2.0, dt=0.2)
        medium = diagnostic(dr=1.0, dt=0.1)
        fine = diagnostic(dr=0.5, dt=0.05)

        err_coarse_medium = abs(medium - coarse)
        err_medium_fine = abs(fine - medium)

        self.assertLess(
            err_medium_fine, err_coarse_medium,
            "Измельчение сетки не уменьшает разницу между решениями - "
            f"|medium-coarse|={err_coarse_medium:.4f}, |fine-medium|={err_medium_fine:.4f}",
        )
        # Оба решения должны быть близки к общему пределу, а не просто монотонно уходить.
        self.assertLess(err_medium_fine / fine, 0.02)


class TestOmegaCaseTable(unittest.TestCase):
    """Регрессия для замечания: build_omega_case_table игнорировал include_comparison."""

    def test_include_comparison_false_returns_single_scenario(self):
        from analysis.scenarios import build_omega_case_table
        table = build_omega_case_table(0.1, 0.3, include_comparison=False)
        self.assertEqual(len(table), 1)

    def test_include_comparison_true_returns_two_distinct_scenarios(self):
        from analysis.scenarios import build_omega_case_table
        table = build_omega_case_table(0.1, 0.3, include_comparison=True)
        self.assertEqual(len(table), 2)
        omega_values = sorted(case["omega_pm"] for case in table.values())
        self.assertEqual(omega_values, [0.1, 0.3])


class TestExcitotoxicExposure(unittest.TestCase):
    """
    Регрессия для новой метрики (см. analysis/excitotoxicity.py): масса за
    заданным радиусом и её накопленная во времени экспозиция - величина,
    напрямую связанная с обоснованием актуальности в курсовой (спилловер к
    внесинаптическим NMDA-R), в отличие от чисто геометрического I(t).
    """

    def test_exposure_matches_manual_trapezoid_integral(self):
        from analysis.excitotoxicity import compute_extrasynaptic_exposure

        r_values = np.array([10.0, 30.0, 50.0])
        t_values = np.array([0.0, 1.0, 3.0])
        # r=50 - единственная точка "вне" (extrasynaptic_radius=40).
        shell_probabilities = np.array([
            [0.6, 0.5, 0.4],  # r=10
            [0.3, 0.2, 0.1],  # r=30
            [0.1, 0.3, 0.5],  # r=50 -> mass_extra
        ])
        result = compute_extrasynaptic_exposure(
            r_values, t_values, shell_probabilities,
            extrasynaptic_radius=40.0,
        )
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result["mass_extra"], [0.1, 0.3, 0.5])
        # Трапеции: [0->1]: 0.5*(0.1+0.3)*1=0.2; [1->3]: 0.5*(0.3+0.5)*2=0.8
        np.testing.assert_allclose(result["exposure_cumulative"], [0.0, 0.2, 1.0])
        self.assertAlmostEqual(result["exposure_total"], 1.0)
        self.assertAlmostEqual(result["peak_mass_extra"], 0.5)
        self.assertAlmostEqual(result["peak_time"], 3.0)

    def test_exposure_none_when_zone_empty(self):
        from analysis.excitotoxicity import compute_extrasynaptic_exposure

        r_values = np.array([10.0, 30.0])
        t_values = np.array([0.0, 1.0])
        shell_probabilities = np.array([[0.5, 0.5], [0.5, 0.5]])
        result = compute_extrasynaptic_exposure(
            r_values, t_values, shell_probabilities,
            extrasynaptic_radius=100.0,
        )
        self.assertIsNone(result)

    def test_exposure_threshold_time_above(self):
        from analysis.excitotoxicity import compute_extrasynaptic_exposure

        r_values = np.array([50.0])
        t_values = np.array([0.0, 1.0, 2.0, 3.0])
        shell_probabilities = np.array([[0.05, 0.2, 0.2, 0.05]])
        result = compute_extrasynaptic_exposure(
            r_values, t_values, shell_probabilities,
            extrasynaptic_radius=0.0,
            concentration_threshold=0.1,
        )
        self.assertIsNotNone(result)
        # Оба конца сегмента [1,2] выше порога 0.1 -> время выше порога = 1.0
        self.assertAlmostEqual(result["time_above_threshold"], 1.0)
        self.assertAlmostEqual(result["first_time_above_threshold"], 1.0)


class TestMassInversionDetector(unittest.TestCase):
    """
    Регрессия для замечания: детектор инверсии должен сравнивать массы
    областей (интеграл p*dV), а не максимумы плотности p.
    """

    def test_find_mass_inversion_time_uses_region_mass(self):
        from analysis.inversion import find_mass_inversion_time
        r_values = np.array([10.0, 30.0, 50.0])
        t_values = np.array([0.0, 1.0])
        # r=10 внутри синапса (<=20), r=30 в переходной зоне (20..40), r=50 - вне.
        shell_probabilities = np.array([
            [0.6, 0.1],  # r=10
            [0.1, 0.6],  # r=30
            [0.3, 0.3],  # r=50
        ])
        result = find_mass_inversion_time(
            r_values, t_values, shell_probabilities,
            synapse_radius=20.0, transition_radius=40.0,
            min_ratio=1.0, pick="first",
        )
        self.assertIsNotNone(result)
        time_val, ratio, mass_tr, mass_syn = result
        self.assertEqual(time_val, 1.0)
        self.assertAlmostEqual(mass_syn, 0.1)
        self.assertAlmostEqual(mass_tr, 0.6)
        self.assertGreater(ratio, 1.0)


if __name__ == "__main__":
    unittest.main()
