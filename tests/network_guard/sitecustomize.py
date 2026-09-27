"""Block external networking in canonical test processes and their children."""

from __future__ import annotations

import ipaddress
import os
import socket


def _loopback_host(host: object) -> bool:
    if host is None:
        return True
    text = str(host).strip().strip("[]")
    if text.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(text).is_loopback
    except ValueError:
        return False


if os.environ.get("CLUSTERLENS_TEST_NO_EXTERNAL_NETWORK") == "1":
    _original_connect = socket.socket.connect
    _original_connect_ex = socket.socket.connect_ex
    _original_getaddrinfo = socket.getaddrinfo

    def _guarded_connect(sock: socket.socket, address) -> None:
        if sock.family == socket.AF_UNIX:
            return _original_connect(sock, address)
        host = address[0] if isinstance(address, tuple) and address else address
        if not _loopback_host(host):
            raise OSError(f"external network disabled by ClusterLens tests: {host}")
        return _original_connect(sock, address)

    def _guarded_connect_ex(sock: socket.socket, address) -> int:
        if sock.family == socket.AF_UNIX:
            return _original_connect_ex(sock, address)
        host = address[0] if isinstance(address, tuple) and address else address
        if not _loopback_host(host):
            raise OSError(f"external network disabled by ClusterLens tests: {host}")
        return _original_connect_ex(sock, address)

    def _guarded_getaddrinfo(host, *args, **kwargs):
        if not _loopback_host(host):
            raise socket.gaierror(f"external DNS disabled by ClusterLens tests: {host}")
        return _original_getaddrinfo(host, *args, **kwargs)

    socket.socket.connect = _guarded_connect
    socket.socket.connect_ex = _guarded_connect_ex
    socket.getaddrinfo = _guarded_getaddrinfo
