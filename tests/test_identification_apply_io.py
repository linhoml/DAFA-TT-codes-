"""ENVI class-map I/O, class filter, and folder listing (no torch)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from identification.io import (  # noqa: E402
    filter_class_map,
    list_input_files,
    write_envi_class_map,
)


class FilterClassMapTests(unittest.TestCase):
    def test_zero_and_empty_keep_all(self):
        src = np.array([[0, 1], [2, 1]], dtype=np.int16)
        np.testing.assert_array_equal(filter_class_map(src, 0), src)
        np.testing.assert_array_equal(filter_class_map(src, ""), src)
        np.testing.assert_array_equal(filter_class_map(src, None), src)

    def test_one_class_only(self):
        src = np.array([[0, 1], [2, 1]], dtype=np.int16)
        shown = filter_class_map(src, 1)
        np.testing.assert_array_equal(shown, np.array([[0, 1], [0, 1]], dtype=np.int16))
        shown2 = filter_class_map(src, "2")
        np.testing.assert_array_equal(shown2, np.array([[0, 0], [2, 0]], dtype=np.int16))


class EnviClassMapTests(unittest.TestCase):
    def test_write_img_and_hdr(self):
        class_map = np.array([[0, 1, 1], [2, 0, 3]], dtype=np.int16)
        names = ["olivine", "pyroxene", "carbonate"]
        with tempfile.TemporaryDirectory() as tmp:
            out = write_envi_class_map(Path(tmp) / "scene_class.img", class_map, names)
            self.assertEqual(out.suffix, ".img")
            self.assertTrue(out.is_file())
            hdr = out.with_suffix(".hdr")
            self.assertTrue(hdr.is_file())
            body = hdr.read_text(encoding="ascii")
            self.assertIn("file type = ENVI Classification", body)
            self.assertIn("data type = 2", body)
            self.assertIn("samples = 3", body)
            self.assertIn("lines = 2", body)
            self.assertIn("olivine", body)
            roundtrip = np.fromfile(out, dtype="<i2").reshape(2, 3)
            np.testing.assert_array_equal(roundtrip, class_map)

    def test_suffix_forced_to_img(self):
        class_map = np.ones((2, 2), dtype=np.int16)
        with tempfile.TemporaryDirectory() as tmp:
            out = write_envi_class_map(Path(tmp) / "map.dat", class_map, ["a"])
            self.assertTrue(out.name.endswith(".img"))
            self.assertTrue(out.with_suffix(".hdr").is_file())


class ListInputFilesTests(unittest.TestCase):
    def test_folder_lists_supported_cubes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.img").write_bytes(b"x")
            (root / "b.mat").write_bytes(b"x")
            (root / "notes.txt").write_text("skip")
            files = list_input_files(root, "*")
            names = {Path(p).name for p in files}
            self.assertEqual(names, {"a.img", "b.mat"})


if __name__ == "__main__":
    unittest.main()
