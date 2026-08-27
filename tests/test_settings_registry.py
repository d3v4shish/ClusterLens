import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from infra.settings import configure_model_cache_environment, get_production_settings_registry


class FakeStore:
    def __init__(self):
        self.values = {}

    def value(self, key, default=None, value_type=None):
        value = self.values.get(key, default)
        return value_type(value) if value_type is not None and value is not None else value

    def setValue(self, key, value):
        self.values[key] = value


def test_registry_validates_cpu_cuda_runtime_choices():
    registry = get_production_settings_registry()
    store = FakeStore()
    store.values["runtime/preferred_mode"] = "directml"

    assert registry.get(store, "runtime/preferred_mode") == "auto"
    assert registry.set(store, "runtime/preferred_mode", "cuda") == "cuda"
    assert store.values["runtime/preferred_mode"] == "cuda"


def test_registry_clamps_batch_worker_cache_and_threshold_values():
    registry = get_production_settings_registry()
    values = registry.validate_values(
        {
            "performance/batch_size_gpu": 0,
            "performance/decode_workers": 999,
            "storage/rebuildable_cache_max_bytes": -1,
            "faces/detector_score_threshold/human": 2.0,
            "updates/checks_enabled": "yes",
        }
    )

    assert values["performance/batch_size_gpu"] == 1
    assert values["performance/decode_workers"] == 128
    assert values["storage/rebuildable_cache_max_bytes"] == 0
    assert values["faces/detector_score_threshold/human"] == 1.0
    assert values["updates/checks_enabled"] is True


def test_registry_defaults_to_scrfd_arcface_gpu_face_pipeline():
    registry = get_production_settings_registry()
    store = FakeStore()

    assert registry.get(store, "faces/default_detector/human") == "scrfd_10g_kps"
    assert registry.get(store, "faces/default_embedder/human") == "arcface_r100_glint360k"
    assert registry.get(store, "faces/detector_score_threshold/human") == 0.35


def test_model_cache_environment_stays_under_the_active_runtime_cache(tmp_path, monkeypatch):
    cache_dir = tmp_path / "runtime" / "cache"
    monkeypatch.setenv("HOME", "/preserved-home")

    configured = configure_model_cache_environment(SimpleNamespace(cache_dir=cache_dir))

    assert configured == cache_dir
    assert os.environ["HF_HOME"] == str(cache_dir / "huggingface")
    assert os.environ["HUGGINGFACE_HUB_CACHE"] == str(cache_dir / "huggingface" / "hub")
    assert os.environ["HF_HUB_CACHE"] == str(cache_dir / "huggingface" / "hub")
    assert os.environ["TORCH_HOME"] == str(cache_dir / "torch")
    assert os.environ["HOME"] == "/preserved-home"
    assert (cache_dir / "huggingface" / "hub").is_dir()
    assert (cache_dir / "torch").is_dir()
