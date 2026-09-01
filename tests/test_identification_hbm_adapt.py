"""HBM cube conversion and class remapping (no scipy / crism_ml train)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from identification.hbm.adapt import (  # noqa: E402
    CRISM_BAND_SELECT,
    cube_to_if_mat,
    remap_prediction,
)
from identification.hbm import ensure_crism_ml  # noqa: E402
from identification.hbm.pipeline import (  # noqa: E402
    _count_classes,
    evaluate_prediction,
    find_trained_model_files,
)


class BandSelectTests(unittest.TestCase):
    def test_select_has_248_channels(self):
        self.assertEqual(len(CRISM_BAND_SELECT), 248)
        self.assertLess(int(CRISM_BAND_SELECT.max()), 438)


class CubeToIfTests(unittest.TestCase):
    def test_438_band_hwb(self):
        cube = np.zeros((6, 8, 438), dtype=np.float32)
        cube[1, 2, 200] = 0.4
        mat = cube_to_if_mat(cube, data_layout="HWB")
        self.assertEqual(mat["IF"].shape, (48, 248))
        self.assertEqual(mat["x"].shape, (48,))
        self.assertEqual(int(mat["y"].max()), 6)
        self.assertEqual(int(mat["x"].max()), 8)

    def test_248_passthrough(self):
        cube = np.ones((2, 3, 248), dtype=np.float32)
        mat = cube_to_if_mat(cube)
        self.assertEqual(mat["IF"].shape, (6, 248))

    def test_wrong_bands_raise(self):
        with self.assertRaises(ValueError):
            cube_to_if_mat(np.zeros((4, 4, 240)))


class RemapTests(unittest.TestCase):
    def test_sparse_codes_become_1_based(self):
        pred = np.array([[0, 14], [33, 14]], dtype=np.int32)
        display, names, codes = remap_prediction(pred)
        self.assertEqual(codes, [14, 33])
        np.testing.assert_array_equal(display, np.array([[0, 1], [2, 1]], dtype=np.int16))
        self.assertEqual(len(names), 2)


class CountClassesTests(unittest.TestCase):
    def test_numpy_class_vector_is_not_used_as_bool(self):
        class _Model:
            classes = np.arange(1, 34)

        self.assertEqual(_count_classes(_Model()), 33)
        self.assertEqual(_count_classes(type("Empty", (), {})()), 0)


class EvaluateTests(unittest.TestCase):
    def test_size_mismatch(self):
        with self.assertRaises(ValueError) as ctx:
            evaluate_prediction(np.ones((4, 4)), np.ones((3, 3)))
        self.assertIn("必须与影像一致", str(ctx.exception))

    def test_oa_perfect(self):
        pred = np.array([[1, 2], [2, 0]])
        lab = np.array([[1, 2], [2, 0]])
        metrics = evaluate_prediction(pred, lab)
        self.assertEqual(metrics["total"], 3)
        self.assertAlmostEqual(metrics["OA"], 1.0)
        self.assertIn("检验精度", metrics["accuracy_report"])


class VendorTests(unittest.TestCase):
    def test_crism_ml_is_vendored(self):
        root = ensure_crism_ml()
        self.assertTrue((root / "crism_ml" / "train.py").is_file())
        self.assertTrue((root / "LICENSE.txt").is_file())


class TrainedModelLookupTests(unittest.TestCase):
    def test_finds_pkls_in_workdir_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            cache.mkdir()
            bland = cache / "default_bmodel.pkl"
            mineral = cache / "default_model.pkl"
            bland.write_bytes(b"bland")
            mineral.write_bytes(b"mineral")
            found_b, found_m = find_trained_model_files(tmp)
            self.assertEqual(found_b.resolve(), bland.resolve())
            self.assertEqual(found_m.resolve(), mineral.resolve())

    def test_missing_models_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as ctx:
                find_trained_model_files(tmp)
            self.assertIn("模型训练", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
