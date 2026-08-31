"""Empty preprocess path must not be treated as the current directory '.'."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from identification.defaults import (  # noqa: E402
    assert_preprocess_model_file,
    find_preprocess_model,
)


class PreprocessPathTests(unittest.TestCase):
    def test_empty_path_is_not_current_directory(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            assert_preprocess_model_file("")
        self.assertNotIn("Permission denied", str(ctx.exception))
        self.assertIn("为空", str(ctx.exception))

        with self.assertRaises(FileNotFoundError):
            assert_preprocess_model_file(".")

        with self.assertRaises(FileNotFoundError):
            assert_preprocess_model_file("./")

    def test_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as ctx:
                assert_preprocess_model_file(tmp)
            self.assertIn("不是有效文件", str(ctx.exception))

    def test_finds_pkl_next_to_training_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkl = root / "preprocess_model.pkl"
            pkl.write_bytes(b"dummy")
            ckpt = root / "result" / "crism" / "crism_seed0_best.pth"
            ckpt.parent.mkdir(parents=True)
            ckpt.write_bytes(b"dummy")
            found = find_preprocess_model(ckpt, "")
            self.assertEqual(found, pkl.resolve())

    def test_explicit_file_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            other = root / "other.pkl"
            other.write_bytes(b"dummy")
            ckpt = root / "model.pth"
            ckpt.write_bytes(b"dummy")
            found = find_preprocess_model(ckpt, other)
            self.assertEqual(found, other.resolve())

    def test_empty_explicit_does_not_match_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(find_preprocess_model(str(Path(tmp) / "missing.pth"), tmp))


if __name__ == "__main__":
    unittest.main()
