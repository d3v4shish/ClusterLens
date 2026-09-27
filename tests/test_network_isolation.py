from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NETWORK_GUARD = REPO_ROOT / "tests" / "network_guard"


def _guarded_environment() -> dict[str, str]:
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(NETWORK_GUARD) + (os.pathsep + existing if existing else "")
    environment["CLUSTERLENS_TEST_NO_EXTERNAL_NETWORK"] = "1"
    return environment


def test_network_guard_blocks_external_dns_in_child_process():
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            (
                "import socket; "
                "\ntry: socket.getaddrinfo('example.com', 443)"
                "\nexcept socket.gaierror as exc: print(exc); raise SystemExit(0)"
                "\nraise SystemExit(9)"
            ),
        ],
        cwd=REPO_ROOT,
        env=_guarded_environment(),
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    assert "external DNS disabled" in completed.stdout


def test_network_guard_allows_loopback_in_child_process():
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            (
                "import socket; "
                "values=socket.getaddrinfo('127.0.0.1', 0, type=socket.SOCK_STREAM); "
                "assert values; print('loopback-ok')"
            ),
        ],
        cwd=REPO_ROOT,
        env=_guarded_environment(),
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "loopback-ok"


def test_network_guard_blocks_external_ip_connection_in_child_process():
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            (
                "import socket; sock=socket.socket(); "
                "\ntry: sock.connect_ex(('203.0.113.1', 443))"
                "\nexcept OSError as exc: print(exc); raise SystemExit(0)"
                "\nraise SystemExit(9)"
            ),
        ],
        cwd=REPO_ROOT,
        env=_guarded_environment(),
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    assert "external network disabled" in completed.stdout
