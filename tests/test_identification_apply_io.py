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
    canonical_match_key,
    classification_stem,
    filter_class_map,
    filename_match_score,
    is_classification_output,
    list_input_files,
    pair_files_by_closest_name,
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
            self.assertIn("data type = 1", body)
            self.assertIn("class lookup = {", body)
            self.assertIn("samples = 3", body)
            self.assertIn("lines = 2", body)
            self.assertIn("olivine", body)
            lookup_line = next(line for line in body.splitlines() if line.startswith("class lookup"))
            rgb = [int(x) for x in lookup_line.split("{", 1)[1].split("}", 1)[0].split(",") if x.strip()]
            self.assertEqual(len(rgb), 4 * 3)
            self.assertEqual(rgb[:3], [0, 0, 0])
            roundtrip = np.fromfile(out, dtype=np.uint8).reshape(2, 3)
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


class ClassificationNameTests(unittest.TestCase):
    def test_stem_from_input_name(self):
        self.assertEqual(
            classification_stem("HRL000040FF_07_IF181L_TRR3.img", "LSGA"),
            "HRL000040FF_07_IF181L_TRR3_LSGA_classification",
        )
        self.assertEqual(
            classification_stem("scene.hdr", "HBM"),
            "scene_HBM_classification",
        )

    def test_does_not_double_suffix(self):
        self.assertEqual(
            classification_stem("scene_LSGA_classification.img", "LSGA"),
            "scene_LSGA_classification",
        )
        self.assertEqual(
            classification_stem("scene_hbm_class", "HBM"),
            "scene_HBM_classification",
        )
        self.assertEqual(
            classification_stem("scene_classification.img", "HBM"),
            "scene_HBM_classification",
        )

    def test_skip_saved_outputs(self):
        self.assertTrue(is_classification_output("scene_LSGA_classification.img"))
        self.assertTrue(is_classification_output("scene_HBM_classification.hdr"))
        self.assertTrue(is_classification_output("scene_classification_codes.hdr"))
        self.assertFalse(is_classification_output("HRL000040FF_07_IF181L_TRR3.img"))


class FilenamePairingTests(unittest.TestCase):
    def test_strips_label_suffixes(self):
        self.assertEqual(canonical_match_key("scene.img"), "scene")
        self.assertEqual(canonical_match_key("scene_label.tif"), "scene")
        self.assertEqual(canonical_match_key("tile_01_gt.mat"), "tile1")
        self.assertEqual(canonical_match_key("tile_1.hdr"), "tile1")

    def test_pairs_closest_unique_names(self):
        cubes = [
            Path("/data/tile_2.img"),
            Path("/data/tile_10.img"),
            Path("/data/scene.hdr"),
        ]
        labels = [
            Path("/lbl/tile_10_label.tif"),
            Path("/lbl/scene_gt.npy"),
            Path("/lbl/tile_2_gt.mat"),
        ]
        pairs = pair_files_by_closest_name(cubes, labels)
        matched = {c.name: lab.name for c, lab, _ in pairs}
        self.assertEqual(matched["tile_2.img"], "tile_2_gt.mat")
        self.assertEqual(matched["tile_10.img"], "tile_10_label.tif")
        self.assertEqual(matched["scene.hdr"], "scene_gt.npy")
        self.assertTrue(all(score == 1.0 for _, _, score in pairs))

    def test_one_cube_picks_closest_of_many_labels(self):
        pairs = pair_files_by_closest_name(
            [Path("FRT0000A123.img")],
            [
                Path("other_gt.tif"),
                Path("FRT0000A123_label.mat"),
                Path("FRT0000B999_label.mat"),
            ],
        )
        self.assertEqual(pairs[0][1].name, "FRT0000A123_label.mat")
        self.assertEqual(pairs[0][2], 1.0)

    def test_too_few_labels_raises(self):
        with self.assertRaises(ValueError):
            pair_files_by_closest_name(
                [Path("a.img"), Path("b.img")],
                [Path("a_label.tif")],
            )

    def test_preserves_cube_order(self):
        cubes = [Path("b.img"), Path("a.img")]
        labels = [Path("a_label.tif"), Path("b_label.tif")]
        pairs = pair_files_by_closest_name(cubes, labels)
        self.assertEqual([c.name for c, _, _ in pairs], ["b.img", "a.img"])
        self.assertGreater(
            filename_match_score("tile_2.img", "tile_2_label.tif"),
            filename_match_score("tile_2.img", "tile_10_label.tif"),
        )


class PairedLabelMosaicTests(unittest.TestCase):
    def test_assembles_per_file_labels(self):
        from identification.crism_common import (  # noqa: E402
            TileMeta,
            assemble_paired_label_map,
            mosaic_shape,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lab_a = np.array([[1, 0], [2, 2]], dtype=np.int64)
            lab_b = np.array([[3, 3], [0, 4]], dtype=np.int64)
            np.save(root / "a_label.npy", lab_a)
            np.save(root / "b_label.npy", lab_b)
            tiles = [
                TileMeta(path=root / "a.img", tile_id=0, start_col=0, width=2, height=2, bands=3),
                TileMeta(path=root / "b.img", tile_id=1, start_col=2, width=2, height=2, bands=3),
            ]
            mosaic = assemble_paired_label_map(
                tiles,
                [root / "a_label.npy", root / "b_label.npy"],
            )
            self.assertEqual(mosaic_shape(tiles), (2, 4))
            self.assertEqual(mosaic.shape, (2, 4))
            np.testing.assert_array_equal(mosaic[0:2, 0:2], lab_a)
            np.testing.assert_array_equal(mosaic[0:2, 2:4], lab_b)

    def test_resolve_label_folder_pairs_by_name(self):
        from identification.crism_common import TileMeta, resolve_tiles_and_label_map

        with tempfile.TemporaryDirectory() as tmp:
            labels = Path(tmp) / "labels"
            labels.mkdir()
            np.save(labels / "alpha_gt.npy", np.array([[1, 2], [0, 0]], dtype=np.int64))
            np.save(labels / "beta_label.npy", np.array([[3, 4], [5, 0]], dtype=np.int64))
            tiles = [
                TileMeta(
                    path=Path("/cubes/alpha.img"),
                    tile_id=0,
                    start_col=99,
                    width=2,
                    height=2,
                    bands=4,
                ),
                TileMeta(
                    path=Path("/cubes/beta.img"),
                    tile_id=1,
                    start_col=99,
                    width=2,
                    height=2,
                    bands=4,
                ),
            ]
            laid, mosaic, mode = resolve_tiles_and_label_map(
                tiles, labels, 600, "tile_id"
            )
            self.assertEqual(mode, "paired")
            self.assertEqual(laid[0].start_col, 0)
            self.assertEqual(laid[1].start_col, 2)
            np.testing.assert_array_equal(
                mosaic,
                np.array([[1, 2, 3, 4], [0, 0, 5, 0]]),
            )

    def test_multi_file_note_lists_order(self):
        from identification.crism_common import (
            TileMeta,
            describe_multi_file_vs_stitched,
        )

        tiles = [
            TileMeta(Path("left.img"), 0, 0, 10, 8, 3),
            TileMeta(Path("right.img"), 1, 10, 12, 8, 3),
        ]
        note = describe_multi_file_vs_stitched(tiles)
        self.assertIn("结果一般不会相同", note)
        self.assertIn("left.img", note)
        self.assertIn("right.img", note)
        self.assertIn("8×22", note)
        self.assertEqual(describe_multi_file_vs_stitched(tiles[:1]), "")

    def test_size_mismatch_raises(self):
        from identification.crism_common import TileMeta, assemble_paired_label_map

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / "a_label.npy", np.ones((3, 3), dtype=np.int64))
            tiles = [
                TileMeta(path=root / "a.img", tile_id=0, start_col=0, width=2, height=2, bands=3),
            ]
            with self.assertRaises(ValueError):
                assemble_paired_label_map(tiles, [root / "a_label.npy"])


if __name__ == "__main__":
    unittest.main()
