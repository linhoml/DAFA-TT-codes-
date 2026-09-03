"""Windowed cube reads so LSGA training does not allocate a 25 GiB mosaic."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from identification.crism_common import TileMeta, iter_tile_windows  # noqa: E402
from identification.io import (  # noqa: E402
    format_cube_memory,
    load_cube_window,
    probe_cube_shape,
    should_load_cube_in_memory,
    _read_envi_window,
)


def _write_envi(root: Path, stem: str, cube: np.ndarray) -> Path:
    lines, samples, bands = cube.shape
    hdr = root / f"{stem}.hdr"
    raster = root / f"{stem}.img"
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


class CubeMemoryTests(unittest.TestCase):
    def test_user_error_shape_is_not_incore(self):
        self.assertFalse(should_load_cube_in_memory(6272, 4497, 240))
        self.assertIn("GiB", format_cube_memory(6272, 4497, 240))
        self.assertTrue(should_load_cube_in_memory(64, 64, 240))

    def test_probe_and_window_envi(self):
        cube = np.arange(6 * 8 * 5, dtype=np.float32).reshape(6, 8, 5)
        with tempfile.TemporaryDirectory() as tmp:
            hdr = _write_envi(Path(tmp), "scene", cube)
            self.assertEqual(probe_cube_shape(hdr), (6, 8, 5))
            self.assertEqual(probe_cube_shape(hdr.with_suffix(".img")), (6, 8, 5))
            window = load_cube_window(hdr, 1, 4, 2, 6)
            np.testing.assert_array_equal(window, cube[1:4, 2:6, :])
            direct = _read_envi_window(hdr, hdr.with_suffix(".img"), 0, 6, 0, 8)
            np.testing.assert_array_equal(direct, cube)

    def test_iter_windows_partition_points(self):
        tile = TileMeta(
            path=Path("big.img"),
            tile_id=0,
            start_col=0,
            width=40,
            height=30,
            bands=240,
        )
        rr, cc = np.meshgrid(np.arange(0, 30, 3), np.arange(0, 40, 5), indexing="ij")
        points = np.stack([rr.ravel(), cc.ravel()], axis=1).astype(np.int64)
        seen = []
        for _r0, _r1, _c0, _c1, pts in iter_tile_windows(tile, points, window=16, halo=4):
            seen.append(pts)
        got = np.concatenate(seen, axis=0)
        got_set = set(map(tuple, got.tolist()))
        src_set = set(map(tuple, points.tolist()))
        self.assertEqual(got_set, src_set)
        self.assertEqual(len(got), len(points))


if __name__ == "__main__":
    unittest.main()
