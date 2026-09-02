"""Label-size checks and evaluation-accuracy log formatting (no torch)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from identification.crism_common import (  # noqa: E402
    TileMeta,
    align_label_to_tiles,
    format_evaluation_report,
)


def _tile(height=4, width=5):
    return [
        TileMeta(
            path=Path("cube.img"),
            tile_id=0,
            start_col=0,
            width=width,
            height=height,
            bands=240,
        )
    ]


class AlignLabelTests(unittest.TestCase):
    def test_matching_shape_passes(self):
        label = np.ones((4, 5), dtype=np.int64)
        aligned = align_label_to_tiles(label, _tile())
        np.testing.assert_array_equal(aligned, label)

    def test_transposed_label_is_accepted(self):
        label = np.arange(20, dtype=np.int64).reshape(5, 4)
        aligned = align_label_to_tiles(label, _tile(4, 5))
        self.assertEqual(aligned.shape, (4, 5))

    def test_mismatch_mentions_required_size(self):
        label = np.ones((3, 3), dtype=np.int64)
        with self.assertRaises(ValueError) as ctx:
            align_label_to_tiles(label, _tile(8, 10))
        message = str(ctx.exception)
        self.assertIn("必须与影像一致", message)
        self.assertIn("(8, 10)", message)
        self.assertIn("(3, 3)", message)


class EvaluationReportTests(unittest.TestCase):
    def test_report_contains_oa_aa_kappa(self):
        metrics = {
            "OA": 0.8,
            "AA": 0.75,
            "Kappa": 0.7,
            "macro_F1": 0.72,
            "total": 10,
            "recall": np.array([0.5, 1.0]),
            "support": np.array([4, 6]),
        }
        text = format_evaluation_report(metrics, ["olivine", "pyroxene"])
        self.assertIn("检验精度", text)
        self.assertIn("OA  = 80.00%", text)
        self.assertIn("AA  = 75.00%", text)
        self.assertIn("Kappa = 0.7000", text)
        self.assertIn("1 olivine: 50.00%", text)
        self.assertIn("2 pyroxene: 100.00%", text)
        self.assertIn("有效标注像元数：10", text)


if __name__ == "__main__":
    unittest.main()
