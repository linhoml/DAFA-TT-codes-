"""Auxiliary cube I/O: PDS .lbl and ordinary ENVI (no Qt / spectral)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from disort.pds_label import load_aux_cube, load_pds_cube  # noqa: E402


def _write_envi(root: Path, stem: str, cube: np.ndarray, raster_ext: str = ".img") -> Path:
    lines, samples, bands = cube.shape
    hdr = root / f"{stem}.hdr"
    raster = root / f"{stem}{raster_ext}"
    hdr.write_text(
        "\n".join(
            [
                "ENVI",
                f"samples = {samples}",
                f"lines = {lines}",
                f"bands = {bands}",
                "header offset = 0",
                "file type = ENVI Standard",
                "data type = 4",
                "interleave = bsq",
                "byte order = 0",
                "",
            ]
        ),
        encoding="ascii",
    )
    np.ascontiguousarray(np.transpose(cube, (2, 0, 1)), dtype=np.float32).tofile(raster)
    return hdr


def _write_pds(root: Path, stem: str, cube: np.ndarray) -> Path:
    lines, samples, bands = cube.shape
    lbl = root / f"{stem}.lbl"
    img = root / f"{stem}.img"
    lbl.write_text(
        "\n".join(
            [
                "PDS_VERSION_ID = PDS3",
                f'^IMAGE = "{stem}.img"',
                "OBJECT = IMAGE",
                f"  LINES = {lines}",
                f"  LINE_SAMPLES = {samples}",
                f"  BANDS = {bands}",
                "  SAMPLE_TYPE = PC_REAL",
                "  SAMPLE_BITS = 32",
                "  BAND_STORAGE_TYPE = BAND_SEQUENTIAL",
                "END_OBJECT = IMAGE",
                "END",
                "",
            ]
        ),
        encoding="ascii",
    )
    np.ascontiguousarray(np.transpose(cube, (2, 0, 1)), dtype=np.float32).tofile(img)
    return lbl


class AuxCubeLoadTests(unittest.TestCase):
    def test_envi_hdr_and_img(self):
        cube = np.arange(2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hdr = _write_envi(root, "geom", cube)
            loaded, meta, header, raster = load_aux_cube(str(hdr))
            np.testing.assert_array_equal(loaded, cube)
            self.assertTrue(header.lower().endswith(".hdr"))
            self.assertTrue(raster.lower().endswith(".img"))
            via_img, *_ = load_aux_cube(str(root / "geom.img"))
            np.testing.assert_array_equal(via_img, cube)
            self.assertEqual(meta["samples"], "3")

    def test_envi_dat_without_lbl(self):
        cube = np.linspace(0.1, 1.5, 2 * 2 * 4, dtype=np.float32).reshape(2, 2, 4)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_envi(root, "aux", cube, raster_ext=".dat")
            loaded, *_ = load_aux_cube(str(root / "aux.dat"))
            np.testing.assert_allclose(loaded, cube)

    def test_pds_lbl_still_works(self):
        cube = np.arange(2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5) + 10
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lbl = _write_pds(root, "ddr", cube)
            loaded, meta, lbl_path, img_path = load_aux_cube(str(lbl))
            np.testing.assert_array_equal(loaded, cube)
            self.assertTrue(lbl_path.lower().endswith(".lbl"))
            self.assertTrue(img_path.lower().endswith(".img"))
            via_pds, *_ = load_pds_cube(str(root / "ddr.img"))
            np.testing.assert_array_equal(via_pds, cube)
            self.assertIn("LINES", meta)

    def test_img_without_header_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            orphan = Path(tmp) / "orphan.img"
            orphan.write_bytes(b"\x00" * 16)
            with self.assertRaises(ValueError):
                load_aux_cube(str(orphan))


if __name__ == "__main__":
    unittest.main()
