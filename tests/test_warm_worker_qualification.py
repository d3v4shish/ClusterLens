from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "qualify_warm_worker.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("qualify_warm_worker", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_real_model_asset_is_not_run(tmp_path):
    qualify = _load_module()

    report, exit_code = qualify.qualify(tmp_path / "missing-model")

    assert exit_code == qualify.EXIT_NOT_RUN
    assert report["status"] == "NOT_RUN"
    assert report["cycles"] == []
    assert report["failures"] == []
    assert report["not_run_reasons"] == ["model.onnx and metadata.json are required"]


def test_model_asset_checksum_mismatch_is_not_run(tmp_path):
    qualify = _load_module()
    asset_dir = tmp_path / "fast_preview"
    asset_dir.mkdir()
    (asset_dir / "model.onnx").write_bytes(b"actual")
    (asset_dir / "metadata.json").write_text(
        '{"model_name":"fast_preview","sha256":"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"}',
        encoding="utf-8",
    )

    report, exit_code = qualify.qualify(asset_dir)

    assert exit_code == qualify.EXIT_NOT_RUN
    assert report["status"] == "NOT_RUN"
    assert "model checksum mismatch" in report["not_run_reasons"][0]


def test_cuda_runtime_unavailable_is_not_run(tmp_path, monkeypatch):
    qualify = _load_module()
    asset_dir = tmp_path / "fast_preview"
    asset_dir.mkdir()
    payload = b"fixture-model"
    (asset_dir / "model.onnx").write_bytes(payload)
    (asset_dir / "metadata.json").write_text(
        '{"model_name":"fast_preview","sha256":"'
        + hashlib.sha256(payload).hexdigest()
        + '"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(qualify, "_cuda_runtime_issue", lambda: "CUDA provider unavailable")

    report, exit_code = qualify.qualify(asset_dir, execution_mode="cuda")

    assert exit_code == qualify.EXIT_NOT_RUN
    assert report["status"] == "NOT_RUN"
    assert report["environment"]["execution_mode"] == "cuda"
    assert report["not_run_reasons"] == ["CUDA provider unavailable"]


def test_nvidia_process_memory_selects_exact_pid(monkeypatch):
    qualify = _load_module()
    monkeypatch.setattr(
        qualify.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="123, 40\n456, 321\ninvalid\n",
        ),
    )

    assert qualify._nvidia_compute_process_memory(456) == 321 * 1024 * 1024
    assert qualify._nvidia_compute_process_memory(999) is None
