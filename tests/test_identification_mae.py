"""Spatial+spectral MAE tokens, mask, and a tiny forward pass."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from identification.io import classification_stem  # noqa: E402

try:
    import torch
    from identification.mae.dataset import (  # noqa: E402
        PrefetchWindowLoader,
        UnlabeledWindowDataset,
        prepare_mae_cube,
    )
    from identification.mae.model import (  # noqa: E402
        MineralMAEClassifier,
        SpatialSpectralEncoder,
        SpatialSpectralMAE,
        diagnostic_spectral_mask,
    )
    from identification.mae.defaults import (  # noqa: E402
        MAE_CROP,
        MAE_SPATIAL_PATCH,
        MAE_SPECTRAL_PATCH,
    )
except ImportError as exc:  # pragma: no cover
    torch = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class MaeNamingTests(unittest.TestCase):
    def test_classification_stem_mae(self):
        self.assertEqual(
            classification_stem("scene", "MAE"),
            "scene_MAE_classification",
        )
        self.assertEqual(
            classification_stem("scene_MAE_classification", "mae"),
            "scene_MAE_classification",
        )


@unittest.skipUnless(torch is not None, f"torch/scipy required: {_IMPORT_ERROR}")
class MaeModelTests(unittest.TestCase):
    def test_diagnostic_patches_cover_hydration_bands(self):
        mask = diagnostic_spectral_mask(15, 16)
        self.assertEqual(tuple(mask.shape), (15,))
        self.assertTrue(bool(mask.any()))
        self.assertFalse(bool(mask.all()))

    def test_patchify_roundtrip(self):
        enc = SpatialSpectralEncoder(depth=1, n_heads=4, d_model=64)
        cube = torch.randn(2, MAE_CROP, MAE_CROP, 240)
        patches = enc.patchify(cube)
        self.assertEqual(
            patches.shape,
            (2, enc.n_patches, MAE_SPATIAL_PATCH * MAE_SPATIAL_PATCH * MAE_SPECTRAL_PATCH),
        )
        back = enc.unpatchify(patches)
        self.assertTrue(torch.allclose(back, cube))

    def test_mae_masked_loss_finite(self):
        enc = SpatialSpectralEncoder(depth=1, n_heads=4, d_model=64)
        mae = SpatialSpectralMAE(
            encoder=enc,
            decoder_dim=32,
            decoder_depth=1,
            decoder_heads=4,
            mask_ratio=0.75,
        )
        cube = torch.rand(2, MAE_CROP, MAE_CROP, 240)
        out = mae(cube)
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertGreater(float(out["n_masked"]), 0)
        keep = int(round(enc.n_patches * 0.25))
        self.assertEqual(int((out["mask"][0] == 0).sum()), keep)

    def test_classifier_tile_map_matches_crop(self):
        enc = SpatialSpectralEncoder(depth=1, n_heads=4, d_model=64)
        clf = MineralMAEClassifier(enc, num_classes=5)
        cube = torch.rand(1, MAE_CROP, MAE_CROP, 240)
        maps, conf = clf.predict_tile(cube)
        self.assertEqual(tuple(maps.shape), (1, MAE_CROP, MAE_CROP))
        self.assertEqual(tuple(conf.shape), (1, MAE_CROP, MAE_CROP))
        self.assertTrue(int(maps.min()) >= 1)
        self.assertTrue(int(maps.max()) <= 5)

    def test_prepare_crop_only_240(self):
        cube = np.random.rand(12, 10, 240).astype(np.float32)
        out = prepare_mae_cube(cube, wavelengths=None, mode="crop")
        self.assertEqual(out.shape[-1], 240)
        norms = np.linalg.norm(out, axis=-1)
        self.assertTrue(np.all(np.isfinite(norms)))

    def test_unlabeled_npy_window(self):
        cube = np.random.rand(40, 36, 240).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scene.npy"
            np.save(path, cube)
            ds = UnlabeledWindowDataset(
                [path],
                crop=32,
                samples_per_epoch=3,
                preprocess_mode="crop",
                seed=1,
            )
            sample = ds[0]
            self.assertEqual(tuple(sample.shape), (32, 32, 240))
            crops = ds.sample_crops(4)
            self.assertEqual(len(crops), 4)
            self.assertEqual(tuple(crops[0].shape), (32, 32, 240))
            loader = PrefetchWindowLoader(
                UnlabeledWindowDataset(
                    [path],
                    crop=32,
                    samples_per_epoch=8,
                    preprocess_mode="crop",
                    seed=2,
                ),
                batch_size=4,
                num_readers=2,
                crops_per_read=2,
                drop_last=True,
            )
            batches = list(loader)
            self.assertEqual(len(batches), 2)
            self.assertEqual(tuple(batches[0].shape), (4, 32, 32, 240))

    def test_empty_envi_img_skipped_then_reads_good_cube(self):
        good = np.random.rand(40, 36, 240).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            npy = root / "good.npy"
            np.save(npy, good)
            hdr = root / "empty.hdr"
            hdr.write_text(
                "\n".join(
                    [
                        "ENVI",
                        "samples = 640",
                        "lines = 420",
                        "bands = 438",
                        "data type = 4",
                        "interleave = bsq",
                        "byte order = 0",
                        "",
                    ]
                ),
                encoding="ascii",
            )
            (root / "empty.img").write_bytes(b"")
            notes = []
            ds = UnlabeledWindowDataset(
                [hdr, npy],
                crop=32,
                samples_per_epoch=2,
                preprocess_mode="crop",
                seed=0,
                log=notes.append,
            )
            self.assertEqual(len(ds.meta), 1)
            self.assertGreaterEqual(ds.skipped, 1)
            self.assertEqual(tuple(ds[0].shape), (32, 32, 240))
            self.assertTrue(any("空" in str(n) or "0 字节" in str(n) for n in notes))

    def test_all_empty_envi_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hdr = root / "empty.hdr"
            hdr.write_text(
                "\n".join(
                    [
                        "ENVI",
                        "samples = 640",
                        "lines = 420",
                        "bands = 438",
                        "data type = 4",
                        "interleave = bsq",
                        "",
                    ]
                ),
                encoding="ascii",
            )
            (root / "empty.img").write_bytes(b"")
            with self.assertRaises(ValueError) as ctx:
                UnlabeledWindowDataset(
                    [hdr],
                    crop=32,
                    samples_per_epoch=2,
                    preprocess_mode="crop",
                )
            self.assertIn("WebDAV", str(ctx.exception))
            self.assertIn("0", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
