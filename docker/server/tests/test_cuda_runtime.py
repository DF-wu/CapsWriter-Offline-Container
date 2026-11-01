import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

SERVER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER))
import download_models as dm


class CudaRuntimeTest(unittest.TestCase):
    def test_cuda_requires_its_backend_library(self):
        self.assertIn("libggml-cuda.so", dm._llama_required_names("cuda"))

    def test_bundled_cuda_installs_verified_libraries_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = {name: ("fixture:" + name).encode() for name in
                     dm._llama_required_names("cpu") + ["libggml-cuda.so"]}
            archive = root / "llama-b7798-cuda-sm61.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                for name, data in files.items():
                    info = tarfile.TarInfo(name); info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
            manifest = root / "runtime.json"
            manifest.write_text(json.dumps({
                "source_commit": "c301172f660a1fe0b42023da990bf7385d69adb4",
                "cuda_architectures": "61",
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "archive_size": archive.stat().st_size,
                "libraries": {k: {"size": len(v), "sha256": hashlib.sha256(v).hexdigest()} for k,v in files.items()},
            }))
            original = Path.cwd()
            try:
                os.chdir(root)
                with patch.dict(dm.LLAMA_CPP_ASSETS), patch.dict(dm.LLAMA_CPP_OFFICIAL_MANIFESTS), patch.object(dm, "LLAMA_TARGET_DIRS", [root / "target"]), patch.dict(os.environ, {"CAPSWRITER_LLAMA_BACKEND": "cuda"}):
                    dm._register_bundled_cuda(manifest)
                    self.assertEqual(dm._prepare_llama_binaries(), 0)
                    self.assertTrue(dm._llama_binaries_ready())
                    (root / "target/libggml-cuda.so").write_bytes(b"corruption")
                    self.assertFalse(dm._llama_binaries_ready())
                    self.assertEqual(dm._prepare_llama_binaries(), 0)
                    self.assertTrue(dm._llama_binaries_ready())
            finally:
                os.chdir(original)

    def run_selection(self, *, nvidia=True, bundle=True, env=None):
        # Run the real selection function with fixture device/artifact paths.
        script = (SERVER / "entrypoint.sh").read_text().split('\nconfigure_backend "$@"')[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            if nvidia: (root / "nvidiactl").touch()
            if bundle: (root / "runtime.json").touch()
            script = script.replace("/dev/nvidiactl", str(root / "nvidiactl")).replace("/dev/dri", str(root / "dri")).replace("/opt/capswriter/llama-cuda/runtime.json", str(root / "runtime.json"))
            script += '\nconfigure_backend\nprintf "RESULT:%s,%s,%s\\n" "$CAPSWRITER_LLAMA_BACKEND" "$CAPSWRITER_QWEN_USE_CUDA" "$CAPSWRITER_QWEN_VULKAN_ENABLE"\n'
            return subprocess.run(["sh", "-c", script], text=True, capture_output=True,
                env={"PATH": os.environ["PATH"], "CAPSWRITER_SETTINGS_BOOTSTRAPPED": "1", **(env or {})})

    def test_nvidia_prefers_bundled_cuda(self):
        result = self.run_selection()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("RESULT:cuda,true,true", result.stdout)

    def test_cpu_low_vram_and_vulkan_overrides(self):
        for env, expected in [
            ({"CAPSWRITER_INFERENCE_HARDWARE": "cpu"}, "cpu,false,false"),
            ({"CAPSWRITER_QWEN_PRESET": "cpu_only"}, "cpu,false,false"),
            ({"CAPSWRITER_QWEN_PRESET": "low_vram_gpu"}, "cpu,true,false"),
            ({"CAPSWRITER_LLAMA_BACKEND": "vulkan"}, "vulkan,true,true"),
            ({"CAPSWRITER_LLAMA_BACKEND": "cpu"}, "cpu,false,false"),
            ({"CAPSWRITER_LLAMA_BACKEND": "cpu", "CAPSWRITER_QWEN_PRESET": "low_vram_gpu"}, "cpu,true,false"),
        ]:
            with self.subTest(env=env):
                result = self.run_selection(env=env)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("RESULT:" + expected, result.stdout)

    def test_old_image_keeps_vulkan_and_no_gpu_keeps_cpu(self):
        self.assertIn("RESULT:vulkan,true,true", self.run_selection(bundle=False).stdout)
        self.assertIn("RESULT:cpu,false,false", self.run_selection(nvidia=False).stdout)

    def test_invalid_backend_fails_clearly(self):
        result = self.run_selection(env={"CAPSWRITER_LLAMA_BACKEND": "typo"})
        self.assertNotEqual(result.returncode, 0)

    def test_cuda_device_guard_rejects_cpu_only_registry(self):
        from fork_server.llama_backend import require_cuda_device
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            require_cuda_device(["CPU"])
        require_cuda_device(["CPU", "CUDA0"])

    def test_selected_cuda_failure_cannot_launch_cpu_service(self):
        script = (SERVER / "entrypoint.sh").read_text().split('\nconfigure_backend "$@"')[0]
        script += '\nexport CAPSWRITER_LLAMA_BACKEND=cuda\nprepare_and_probe_cpu_fallback\n'
        result = subprocess.run(["sh", "-c", script], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing CPU fallback", result.stderr)

    def test_engine_packages_propagate_cuda_import_failures(self):
        import types
        def failing_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "os":
                return os
            if name == "llama":
                raise RuntimeError("CUDA registry fixture failure")
            return types.SimpleNamespace(**{key: None for key in fromlist})
        for engine in ['qwen_asr_gguf', 'fun_asr_gguf', 'force_aligner_gguf']:
            path = SERVER.parents[1] / 'core/server/engines' / engine / 'inference/__init__.py'
            with self.subTest(engine=engine), patch.dict(os.environ, {"CAPSWRITER_LLAMA_BACKEND": "cuda"}):
                with self.assertRaisesRegex(RuntimeError, "CUDA registry fixture failure"):
                    exec(path.read_text(), {"__builtins__": {"__import__": failing_import, "Exception": Exception}})

if __name__ == "__main__":
    unittest.main()
