"""Device selection must not silently fall back to CPU when the user asked for CUDA."""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))


def _install_fake_torch(*, available: bool, count: int = 0, cuda_built=None):
    torch = types.ModuleType("torch")
    cuda = types.ModuleType("torch.cuda")
    version = types.ModuleType("torch.version")
    version.cuda = cuda_built

    class Device:
        def __init__(self, spec):
            spec = str(spec)
            if spec.startswith("cuda"):
                self.type = "cuda"
                self.index = int(spec.split(":", 1)[1]) if ":" in spec else 0
            else:
                self.type = "cpu"
                self.index = None

        def __repr__(self):
            if self.type == "cpu":
                return "device(type='cpu')"
            return f"device(type='cuda', index={self.index})"

    torch.device = Device
    torch.__version__ = "2.4.0+cpu"
    torch.__file__ = "/fake/lib/python/site-packages/torch/__init__.py"
    torch.cuda = cuda
    torch.version = version
    cuda.is_available = lambda: available
    cuda.device_count = lambda: count
    cuda.get_device_name = lambda _i: "Fake NVIDIA GPU"
    sys.modules["torch"] = torch
    sys.modules["torch.cuda"] = cuda
    sys.modules["torch.version"] = version
    return torch


_install_fake_torch(available=False, count=0, cuda_built=None)

from identification.crism_common import (  # noqa: E402
    cpu_device_warning,
    cuda_unavailable_message,
    is_cuda_request,
    python_executable,
    resolve_device,
)


class ResolveDeviceTests(unittest.TestCase):
    def setUp(self):
        _install_fake_torch(available=False, count=0, cuda_built=None)

    def test_cpu_string(self):
        device = resolve_device("cpu")
        self.assertEqual(device.type, "cpu")

    def test_empty_string_is_cpu(self):
        device = resolve_device("  ")
        self.assertEqual(device.type, "cpu")

    def test_negative_index_is_cpu(self):
        device = resolve_device(-1)
        self.assertEqual(device.type, "cpu")

    def test_cuda_request_detection(self):
        self.assertTrue(is_cuda_request("cuda:0"))
        self.assertTrue(is_cuda_request("CUDA"))
        self.assertTrue(is_cuda_request(0))
        self.assertFalse(is_cuda_request("cpu"))
        self.assertFalse(is_cuda_request(-1))

    def test_explicit_cuda_does_not_silently_use_cpu(self):
        with self.assertRaises(RuntimeError) as ctx:
            resolve_device("cuda:0")
        message = str(ctx.exception)
        self.assertIn("cuda.is_available()", message)
        self.assertIn("CPU 版", message)
        self.assertNotIn("请把设备改成 cuda:0", message)
        self.assertIn(python_executable(), message)
        self.assertIn("-m pip install", message)

    def test_integer_gpu_index_does_not_silently_use_cpu(self):
        with self.assertRaises(RuntimeError):
            resolve_device(0)

    def test_cuda_available_uses_gpu(self):
        _install_fake_torch(available=True, count=1, cuda_built="12.8")
        device = resolve_device("cuda:0")
        self.assertEqual(device.type, "cuda")
        self.assertEqual(device.index, 0)

    def test_cpu_warning_does_not_tell_user_to_pick_cuda_if_already_picked(self):
        text = cpu_device_warning("cuda:0")
        self.assertNotIn("请把设备改成 cuda:0", text)
        self.assertNotIn("请把计算设备改成 cuda:0", text)
        self.assertIn("CPU 版", text)

    def test_cpu_warning_for_explicit_cpu(self):
        text = cpu_device_warning("cpu")
        self.assertIn("计算设备是 CPU", text)
        self.assertIn("cuda:0", text)

    def test_python_executable_is_this_interpreter(self):
        self.assertEqual(python_executable(), sys.executable)
        text = cuda_unavailable_message("cuda:0")
        self.assertIn(sys.executable, text)


if __name__ == "__main__":
    unittest.main()
