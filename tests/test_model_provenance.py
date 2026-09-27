from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from app.services.clustering_options import clustering_model_names  # noqa: E402
from app.services.model_assets import (  # noqa: E402
    HF_MODEL_REPOSITORIES,
    HF_MODEL_REVISIONS,
    MODEL_LICENSES,
    MODEL_SOURCE_URLS,
)
from ml.embeddings import ModelManager  # noqa: E402


def test_production_model_catalog_has_explicit_source_and_license_status() -> None:
    for model_name in clustering_model_names(scope="production"):
        assert MODEL_SOURCE_URLS.get(model_name, "").startswith("https://")
        license_text = MODEL_LICENSES.get(model_name, "")
        assert license_text
        assert "verify" not in license_text.lower()


def test_managed_huggingface_revisions_are_full_immutable_commits() -> None:
    assert set(HF_MODEL_REPOSITORIES) == set(HF_MODEL_REVISIONS)
    assert {"mobileclip", "dino", "dinov2_base", "clip", "openclip", "siglip"} <= set(
        HF_MODEL_REVISIONS
    )
    assert all(re.fullmatch(r"[0-9a-f]{40}", revision) for revision in HF_MODEL_REVISIONS.values())


def test_runtime_loaders_use_the_same_pinned_revision_as_downloads() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class _FakeLoader:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs):
            calls.append((model_id, kwargs))
            return torch.nn.Identity()

    fake_transformers = SimpleNamespace(
        AutoImageProcessor=_FakeLoader,
        AutoTokenizer=_FakeLoader,
        CLIPModel=_FakeLoader,
        SiglipModel=_FakeLoader,
    )
    manager = object.__new__(ModelManager)
    manager.allow_model_downloads = True
    manager.load_text_tokenizer = True
    manager.execution_policy = SimpleNamespace(
        effective_mode="cpu",
        torch_device="cpu",
        onnx_provider="CPUExecutionProvider",
    )
    manager.device = torch.device("cpu")
    manager.model_asset_service = SimpleNamespace(
        find_bundle=lambda _model_name: None,
        local_cache_present=lambda _model_name: True,
        local_cache_revision=lambda _model_name: "fixture-cache-revision",
    )
    manager.onnx_service = SimpleNamespace()

    with patch.dict(sys.modules, {"transformers": fake_transformers}):
        for model_name in ("clip", "openclip", "siglip"):
            calls.clear()
            bundle = manager._load_bundle(model_name, use_onnx=False)
            assert bundle.model_name == model_name
            assert len(calls) == 3
            assert {model_id for model_id, _kwargs in calls} == {
                HF_MODEL_REPOSITORIES[model_name]
            }
            assert {
                kwargs.get("revision") for _model_id, kwargs in calls
            } == {HF_MODEL_REVISIONS[model_name]}

        manager.allow_model_downloads = False
        calls.clear()
        manager._load_bundle("clip", use_onnx=False)
        assert len(calls) == 3
        assert all("revision" not in kwargs for _model_id, kwargs in calls)
        assert all(kwargs.get("local_files_only") is True for _model_id, kwargs in calls)


def test_timm_runtime_loaders_pin_online_and_preserve_local_cache_recovery() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def create_model(model_id: str, **kwargs):
        calls.append((model_id, kwargs))
        return torch.nn.Identity()

    manager = object.__new__(ModelManager)
    manager.allow_model_downloads = True
    manager.load_text_tokenizer = False
    manager.execution_policy = SimpleNamespace(
        effective_mode="cpu",
        torch_device="cpu",
        onnx_provider="CPUExecutionProvider",
    )
    manager.device = torch.device("cpu")
    manager.model_asset_service = SimpleNamespace(
        find_bundle=lambda _model_name: None,
        local_cache_present=lambda _model_name: True,
        local_cache_revision=lambda _model_name: "fixture-cache-revision",
    )
    manager.onnx_service = SimpleNamespace()
    manager._timm_preprocess = lambda _model, *, fallback_input_size: (
        object(),
        fallback_input_size,
    )

    with patch.dict(sys.modules, {"timm": SimpleNamespace(create_model=create_model)}):
        for model_name, timm_model_id in (
            ("dino", "vit_small_patch16_224_dino"),
            ("dinov2_base", "vit_base_patch14_dinov2"),
        ):
            calls.clear()
            manager._load_bundle(model_name, use_onnx=False)
            assert calls == [
                (
                    timm_model_id,
                    {
                        "pretrained": True,
                        "num_classes": 0,
                        "pretrained_cfg_overlay": {
                            "hf_hub_id": (
                                f"{HF_MODEL_REPOSITORIES[model_name]}@"
                                f"{HF_MODEL_REVISIONS[model_name]}"
                            )
                        },
                    },
                )
            ]

        calls.clear()
        manager._load_bundle("mobileclip", use_onnx=False)
        assert calls == [
            (
                (
                    f"hf_hub:{HF_MODEL_REPOSITORIES['mobileclip']}@"
                    f"{HF_MODEL_REVISIONS['mobileclip']}"
                ),
                {"pretrained": True, "num_classes": 0},
            )
        ]

        manager.allow_model_downloads = False
        for model_name in ("dino", "dinov2_base", "mobileclip"):
            calls.clear()
            manager._load_bundle(model_name, use_onnx=False)
            assert len(calls) == 1
            model_id, kwargs = calls[0]
            assert HF_MODEL_REVISIONS[model_name] not in model_id
            assert "pretrained_cfg_overlay" not in kwargs


def test_face_catalog_has_source_and_unambiguous_license_policy() -> None:
    metadata_paths = sorted((REPO_ROOT / "face_model_assets").rglob("metadata.json"))
    assert len(metadata_paths) == 14
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert str(metadata.get("source_url") or "").startswith("https://")
        license_text = str(metadata.get("license") or "")
        assert license_text
        assert "verify" not in license_text.lower()
