# -*- coding: utf-8 -*-
"""
tests/test_geometry_mixed.py - этап 1: смешанная геометрия Barbour (2001) и источник ровно N молекул.

Запуск: python -m unittest tests.test_geometry_mixed -v
"""
import math
import unittest

import numpy as np

from tests._physical_helpers import cfg, config, run_physical  # noqa: F401
from solver.geometry import barbour_area, barbour_volume, radial_cell_measures, radial_face_measures
from physics.pulses import apply_pulse
from utils.units import MOLECULES_PER_NM3_TO_UM, solver_to_uM_factor

A_NM, B_NM, H, ALPHA = 200.0, 400.0, 20.0, 0.2


def _geo():
    return cfg(geometry_a_nm=A_NM, geometry_b_nm=B_NM, cleft_height_nm=H, extracellular_volume_fraction=ALPHA)


class TestBarbourGeometry(unittest.TestCase):
    def test_volume_matches_cleft_and_neuropil_laws(self):
        with _geo():
            r_in = np.array([10.0, 100.0, 199.0])
            r_out = np.array([450.0, 800.0, 2000.0])
            np.testing.assert_allclose(barbour_volume(r_in), math.pi * H * r_in**2, rtol=1e-13)
            np.testing.assert_allclose(barbour_volume(r_out), ALPHA * 4.0 / 3.0 * math.pi * r_out**3, rtol=1e-13)

    def test_area_positive_and_equals_dV_dr(self):
        with _geo():
            r = np.linspace(1.0, 1500.0, 30001)
            A = barbour_area(r)
            self.assertTrue(np.all(A > 0.0))
            dVdr = np.gradient(barbour_volume(r), r)
            rel = np.abs(dVdr[2:-2] - A[2:-2]) / A[2:-2]
            self.assertLess(rel.max(), 1e-5)

    def test_area_and_its_slope_continuous_at_a_and_b(self):
        """s(x) = 6x^5-15x^4+10x^3: значения и наклон A(r) непрерывны на стыках a и b."""
        eps = 1e-6
        with _geo():
            for x in (A_NM, B_NM):
                left, right = barbour_area(x - eps), barbour_area(x + eps)
                self.assertLess(abs(right - left) / left, 1e-6)
                sl = (barbour_area(x - eps) - barbour_area(x - 1e-3)) / (1e-3 - eps)
                sr = (barbour_area(x + 1e-3) - barbour_area(x + eps)) / (1e-3 - eps)
                self.assertLess(abs(sr - sl) / max(abs(sl), 1.0), 1e-3)

    def test_cell_volumes_sum_to_V_of_R(self):
        with _geo():
            dr, R = 2.0, 3000.0
            r = (np.arange(int(R / dr)) + 0.5) * dr
            dV = radial_cell_measures(r, dr, "barbour_mixed")
            self.assertTrue(np.all(dV > 0.0))
            self.assertAlmostEqual(dV.sum() / barbour_volume(R), 1.0, places=12)
            faces = radial_face_measures(np.array([A_NM, B_NM]), "barbour_mixed")
            np.testing.assert_allclose(faces, barbour_area(np.array([A_NM, B_NM])))

    def test_unit_factor_has_no_alpha(self):
        """В физической ветке c уже на нм³ жидкости: перевод в мкМ без деления на α."""
        for alpha in (0.1, 0.2, 0.5):
            self.assertEqual(solver_to_uM_factor(alpha, "barbour_mixed", H), MOLECULES_PER_NM3_TO_UM)

    def test_pulse_adds_exactly_N_molecules(self):
        with _geo(), cfg(pulse_sigma_nm=20.0, pulse_center_nm=0.0):
            dr = 2.0
            r = (np.arange(1500) + 0.5) * dr
            dV = radial_cell_measures(r, dr, "barbour_mixed")
            c = apply_pulse(np.zeros_like(r), r, (0.0, 100.0), 3000.0, dr, "barbour_mixed", renormalize=False)
            self.assertAlmostEqual(float(np.sum(c * dV)), 3000.0, places=8)

    def test_cleft_concentration_depends_on_h_not_on_alpha(self):
        """Пик в щели сразу после выброса: N/(объём диска) - зависит от h, не от α."""
        peaks = {}
        for h in (15.0, 20.0):
            for alpha in (0.15, 0.25):
                with cfg(geometry_a_nm=A_NM, geometry_b_nm=B_NM, cleft_height_nm=h,
                         extracellular_volume_fraction=alpha, pulse_sigma_nm=20.0, pulse_center_nm=0.0):
                    dr = 1.0
                    r = (np.arange(1000) + 0.5) * dr
                    c = apply_pulse(np.zeros_like(r), r, (0.0, 100.0), 3000.0, dr, "barbour_mixed", renormalize=False)
                    peaks[(h, alpha)] = c[0]
        self.assertAlmostEqual(peaks[(20.0, 0.15)] / peaks[(20.0, 0.25)], 1.0, places=12)
        self.assertAlmostEqual(peaks[(15.0, 0.15)] / peaks[(20.0, 0.15)], 20.0 / 15.0, places=10)


class TestAnalyticDiffusion(unittest.TestCase):
    """Совпадение с аналитическим решением до прихода фронта к границе."""

    def test_3d_porous_medium_gaussian(self):
        # a, b малы -> почти вся область 3D-нейропиль. Гаусс σ0: c = N/(α(2πs²)^1.5)·exp(-r²/2s²), s² = σ0² + 2Dt
        N, D, s0, T = 3000.0, 320.0, 60.0, 400.0
        c, r, t, info = run_physical(D_cleft=D, D_pm=D, a=0.5, b=1.0, sigma=s0, window=6 * s0,
                                     R=4000.0, dr=2.0, dt=0.25, T=T, boundary="no_flux")
        s2 = s0**2 + 2.0 * D * t[-1]
        exact = N / (ALPHA * (2.0 * math.pi * s2) ** 1.5) * np.exp(-r**2 / (2.0 * s2))
        m = r < 2.0 * math.sqrt(s2)
        rel = np.abs(c[m, -1] - exact[m]) / exact[m].max()
        self.assertLess(rel.max(), 0.02)

    def test_2d_disk_gaussian(self):
        # a, b больше области -> вся область - щель-диск. c = N/(h·2πs²)·exp(-r²/2s²)
        N, D, s0, T = 3000.0, 330.0, 60.0, 400.0
        c, r, t, info = run_physical(D_cleft=D, D_pm=D, a=5000.0, b=6000.0, sigma=s0, window=6 * s0,
                                     R=4000.0, dr=2.0, dt=0.25, T=T, boundary="no_flux")
        s2 = s0**2 + 2.0 * D * t[-1]
        exact = N / (H * 2.0 * math.pi * s2) * np.exp(-r**2 / (2.0 * s2))
        m = r < 2.0 * math.sqrt(s2)
        rel = np.abs(c[m, -1] - exact[m]) / exact[m].max()
        self.assertLess(rel.max(), 0.02)

    def test_outer_radius_does_not_change_near_field(self):
        """Концентрация у синапса (r <= 2 мкм, t = 1 мс) не зависит от положения границы R = 3, 6, 9 мкм."""
        probes = []
        for R in (3000.0, 6000.0, 9000.0):
            c, r, t, info = run_physical(R=R, dr=4.0, dt=1.0, T=1000.0, frames=50)
            probes.append(np.interp([200.0, 500.0, 1000.0, 2000.0], r, c[:, -1]))
        probes = np.array(probes)
        self.assertLess(np.max(np.abs(probes[1] - probes[2]) / probes[2]), 1e-3)
        self.assertLess(np.max(np.abs(probes[0] - probes[2]) / probes[2]), 0.05)


if __name__ == "__main__":
    unittest.main()
