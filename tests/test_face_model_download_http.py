from __future__ import annotations

import hashlib
import socket
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services.face_model_installer import FaceModelInstaller
from infra.cancel import Cancelled


class _DownloadHandler(BaseHTTPRequestHandler):
    server_version = "ClusterLensTestHTTP/1"

    def do_GET(self):  # noqa: N802 - stdlib handler contract
        state = self.server.state
        payload = state["payload"]
        range_header = self.headers.get("Range", "")
        state["requests"].append(range_header)
        start = int(range_header.removeprefix("bytes=").removesuffix("-") or 0) if range_header else 0
        body = payload[start:]
        self.send_response(206 if range_header else 200)
        self.send_header("Content-Length", str(len(body)))
        if range_header:
            self.send_header("Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}")
        self.end_headers()
        if state["disconnect_first"] and len(state["requests"]) == 1:
            cutoff = min(256 * 1024, len(body))
            self.wfile.write(body[:cutoff])
            self.wfile.flush()
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.close_connection = True
            return
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def log_message(self, _format, *_args):
        return


@contextmanager
def _download_server(payload: bytes, *, disconnect_first: bool = False):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _DownloadHandler)
    server.state = {
        "payload": payload,
        "disconnect_first": disconnect_first,
        "requests": [],
    }
    thread = threading.Thread(target=server.serve_forever, name="face-model-test-http", daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}/model.bin", server.state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _installer(tmp_path: Path) -> FaceModelInstaller:
    return FaceModelInstaller(settings=SimpleNamespace(cache_dir=tmp_path))


def test_local_http_interruption_retries_with_range_and_reuses_verified_cache(tmp_path):
    payload = bytes(range(256)) * 4096
    expected_sha = hashlib.sha256(payload).hexdigest()
    installer = _installer(tmp_path)

    with _download_server(payload, disconnect_first=True) as (url, state):
        first_path, first_temp = installer._download_to_temp(
            url,
            expected_sha,
            "model.bin",
            progress_prefix="Downloading local fixture",
        )
        second_path, second_temp = installer._download_to_temp(
            url,
            expected_sha,
            "model.bin",
            progress_prefix="Downloading local fixture",
        )

    assert first_path.read_bytes() == payload
    assert second_path.read_bytes() == payload
    assert state["requests"] == ["", "bytes=262144-"]
    assert len(list((tmp_path / "face_model_downloads").glob("*.verified.json"))) == 1
    assert Path(first_temp).is_dir()
    assert Path(second_temp).is_dir()


def test_local_http_checksum_failure_never_promotes_payload(tmp_path):
    payload = b"corrupt-model-payload"
    installer = _installer(tmp_path)

    with _download_server(payload) as (url, state), pytest.raises(RuntimeError, match="Checksum mismatch"):
        installer._download_to_temp(
            url,
            hashlib.sha256(b"expected-model-payload").hexdigest(),
            "model.bin",
            progress_prefix="Downloading corrupt fixture",
        )

    cache_entries = list((tmp_path / "face_model_downloads").iterdir())
    assert state["requests"] == [""]
    assert cache_entries == []


def test_local_http_cancellation_keeps_partial_for_safe_resume(tmp_path):
    payload = bytes(range(251)) * 5000
    expected_sha = hashlib.sha256(payload).hexdigest()
    installer = _installer(tmp_path)
    download_cache = installer.download_cache_dir()

    def cancel_after_first_chunk() -> bool:
        return any(path.stat().st_size > 0 for path in download_cache.glob("*.partial"))

    with _download_server(payload) as (url, state):
        with pytest.raises(Cancelled):
            installer._download_to_temp(
                url,
                expected_sha,
                "model.bin",
                cancel_check=cancel_after_first_chunk,
                progress_prefix="Downloading cancellable fixture",
            )
        partials = list(download_cache.glob("*.partial"))
        assert len(partials) == 1
        partial_size = partials[0].stat().st_size
        assert 0 < partial_size < len(payload)

        resumed_path, _temp_root = installer._download_to_temp(
            url,
            expected_sha,
            "model.bin",
            progress_prefix="Resuming local fixture",
        )

    assert resumed_path.read_bytes() == payload
    assert state["requests"] == ["", f"bytes={partial_size}-"]
    assert not list(download_cache.glob("*.partial"))
