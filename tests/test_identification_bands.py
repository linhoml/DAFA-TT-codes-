"""Wavelength crop for identification uses 1.02–2.6 μm."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from identification.bands import (  # noqa: E402
    cube_to_identification_range,
    crism_target_wavelengths,
)
from identification.defaults import APPLY_WL_MAX, APPLY_WL_MIN, TARGET_BAND_NUM


class IdentificationBandTests(unittest.TestCase):
    def test_target_grid(self):
        wl = crism_target_wavelengths()
        self.assertEqual(wl.size, TARGET_BAND_NUM)
        self.assertAlmostEqual(float(wl[0]), APPLY_WL_MIN, places=6)
        self.assertAlmostEqual(float(wl[-1]), APPLY_WL_MAX, places=6)
        self.assertEqual(APPLY_WL_MAX, 2.6)

    def test_crop_by_wavelength(self):
        height, width, bands = 4, 5, 50
        wl = np.linspace(0.8, 3.0, bands)
        cube = np.zeros((height, width, bands), dtype=np.float32)
        cube[..., :] = np.arange(bands, dtype=np.float32)
        out, out_wl = cube_to_identification_range(cube, wl)
        self.assertEqual(out.shape, (height, width, TARGET_BAND_NUM))
        self.assertAlmostEqual(float(out_wl[0]), APPLY_WL_MIN, places=5)
        self.assertAlmostEqual(float(out_wl[-1]), APPLY_WL_MAX, places=5)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_438_without_wavelengths_uses_crism_slice(self):
        cube = np.random.rand(3, 4, 438).astype(np.float32)
        out, out_wl = cube_to_identification_range(cube, None)
        self.assertEqual(out.shape[-1], TARGET_BAND_NUM)
        np.testing.assert_allclose(out, cube[:, :, 3:243])
        self.assertAlmostEqual(float(out_wl[-1]), APPLY_WL_MAX, places=5)

    def test_wavelengths_in_nm_are_converted(self):
        bands = 30
        wl_nm = np.linspace(1000.0, 2700.0, bands)
        cube = np.ones((2, 2, bands), dtype=np.float32)
        out, out_wl = cube_to_identification_range(cube, wl_nm)
        self.assertEqual(out.shape[-1], TARGET_BAND_NUM)
        self.assertLess(float(out_wl[0]), 2.0)


if __name__ == "__main__":
    unittest.main()
