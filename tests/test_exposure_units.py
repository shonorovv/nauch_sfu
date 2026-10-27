# -*- coding: utf-8 -*-
"""
tests/test_exposure_units.py - P0.5 и P1.4 ТЗ от 13.10.2026: единицы метрик экспозиции и энтропии.

Запуск: python -m unittest tests.test_exposure_units -v
"""
import unittest

import numpy as np

from tests._physical_helpers import cfg
from analysis.extrasynaptic_exposure import compute_local_concentration_exposure, compute_mass_exposure
from analysis.entropy import compute_local_entropy_fields, conditional_probability
from solver.geometry import radial_cell_measures


class TestMassExposure(unittest.TestCase):
    def test_integral_has_molecule_time_units(self):
        r = np.array([100.0, 500.0, 900.0]); t = np.array([0.0, 2.0, 4.0])
        q = np.array([[5.0, 5.0, 5.0], [1.0, 3.0, 1.0], [1.0, 1.0, 1.0]])    # молекулы в оболочках
        res = compute_mass_exposure(r, t, q, 400.0)
        self.assertTrue(np.allclose(res["mass_extra"], [2.0, 4.0, 2.0]))
        self.assertAlmostEqual(res["exposure_total"], 12.0)                  # (2+4)/2·2 + (4+2)/2·2

    def test_mass_threshold_is_not_a_concentration(self):
        r = np.array([100.0, 500.0]); t = np.array([0.0, 1.0, 2.0])
        q = np.array([[9.0, 9.0, 9.0], [2.0, 2.0, 0.0]])
        res = compute_mass_exposure(r, t, q, 400.0, mass_threshold=1.0)
        self.assertAlmostEqual(res["time_above_mass_threshold"], 1.0)


class TestLocalConcentrationExposure(unittest.TestCase):
    def test_threshold_applies_to_local_concentration(self):
        r = np.array([0.0, 500.0]); t = np.array([0.0, 1.0, 2.0, 3.0])
        c = np.array([[100.0] * 4, [0.0, 2.0, 2.0, 0.0]])                    # мкМ
        res = compute_local_concentration_exposure(c, r, t, 500.0, 1.0)
        self.assertAlmostEqual(res["time_above_threshold_us"], 2.0)          # 0.5 + 1 + 0.5 с интерполяцией
        self.assertAlmostEqual(res["excess_integral_uM_us"], 1.5)           # ∫(c − 1) dt, мкМ·мкс: 0,25 + 1 + 0,25
        self.assertAlmostEqual(res["peak_uM"], 2.0)

    def test_never_above(self):
        r = np.array([500.0]); t = np.array([0.0, 1.0]); c = np.array([[0.1, 0.1]])
        res = compute_local_concentration_exposure(c, r, t, 500.0, 1.0)
        self.assertEqual(res["time_above_threshold_us"], 0.0)
        self.assertIsNone(res["first_time_above_threshold_us"])


class TestConditionalEntropy(unittest.TestCase):
    def setUp(self):
        self.dr = 2.0
        self.r = (np.arange(50) + 0.5) * self.dr
        self.p = np.exp(-self.r[:, None] ** 2 / (2 * 20.0 ** 2)) * np.array([[1.0, 0.5]])  # масса падает вдвое

    def test_conditional_probability_has_unit_mass(self):
        q = conditional_probability(self.p, self.r, self.dr)
        m = radial_cell_measures(self.r, self.dr)[:, None]
        self.assertTrue(np.allclose((q * m).sum(0), 1.0))

    def test_concentration_mode_is_scale_invariant(self):
        with cfg(normalization_mode="concentration"):
            f1 = compute_local_entropy_fields(self.p, self.r, self.dr)
            f2 = compute_local_entropy_fields(self.p * 1000.0, self.r, self.dr)
        self.assertTrue(f1["conditional_normalization"])
        self.assertTrue(np.allclose(f1["shannon_shell"], f2["shannon_shell"]))
        self.assertTrue(np.allclose(f1["shell_molecules"].sum(0) * 1000.0, f2["shell_molecules"].sum(0)))

    def test_probability_mode_unchanged(self):
        with cfg(normalization_mode="probability"):
            f = compute_local_entropy_fields(self.p, self.r, self.dr)
        self.assertFalse(f["conditional_normalization"])


if __name__ == "__main__":
    unittest.main()
