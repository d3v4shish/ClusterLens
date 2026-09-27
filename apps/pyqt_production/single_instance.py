"""Local, per-data-home single-instance activation for the desktop shell."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QObject
from PyQt6.QtNetwork import QLocalServer, QLocalSocket


class SingleInstanceCoordinator(QObject):
    """Own a local activation endpoint, or hand the launch to its owner.

    The endpoint includes the canonical Data Home.  This keeps two deliberately
    isolated ClusterLens profiles independent while preventing accidental
    duplicate windows for the same profile.
    """

    _REQUEST = b"activate\n"
    _REPLY = b"ok\n"
    _CONNECT_TIMEOUT_MS = 350

    def __init__(self, runtime_root: str | Path, parent=None) -> None:
        super().__init__(parent)
        try:
            canonical_root = Path(runtime_root).expanduser().resolve()
        except OSError:
            canonical_root = Path(runtime_root).expanduser().absolute()
        digest = hashlib.sha256(str(canonical_root).encode("utf-8")).hexdigest()[:20]
        self.server_name = f"clusterlens-{digest}"
        self._server = QLocalServer(self)
        self._activate: Callable[[], None] | None = None
        self._owns_server = False

    def acquire_or_handoff(self, activate: Callable[[], None]) -> bool:
        """Return true only when this process becomes the primary instance."""

        self._activate = activate
        if self._handoff_to_primary():
            return False
        if self._listen():
            return True

        # A concurrent primary might have won the race.  Prefer it whenever it
        # answers before considering cleanup of an abandoned local socket.
        if self._handoff_to_primary():
            return False
        QLocalServer.removeServer(self.server_name)
        if self._listen():
            return True
        raise RuntimeError(f"Unable to create ClusterLens activation endpoint: {self.server_name}")

    def set_activation_handler(self, activate: Callable[[], None]) -> None:
        """Replace the primary handler after its main window is constructed."""

        self._activate = activate

    def close(self) -> None:
        """Release only the endpoint this coordinator successfully owned."""

        if not self._owns_server:
            return
        self._server.close()
        QLocalServer.removeServer(self.server_name)
        self._owns_server = False

    def _listen(self) -> bool:
        if not self._server.listen(self.server_name):
            return False
        self._owns_server = True
        self._server.newConnection.connect(self._accept_connections)
        return True

    def _handoff_to_primary(self) -> bool:
        socket = QLocalSocket(self)
        try:
            socket.connectToServer(self.server_name)
            if not socket.waitForConnected(self._CONNECT_TIMEOUT_MS):
                return False
            socket.write(self._REQUEST)
            socket.flush()
            # A connected server is sufficient evidence of a primary.  The
            # acknowledgement is a best-effort confirmation so an occupied UI
            # thread cannot accidentally cause a second launch.
            socket.waitForReadyRead(self._CONNECT_TIMEOUT_MS)
            return True
        finally:
            socket.disconnectFromServer()
            socket.deleteLater()

    def _accept_connections(self) -> None:
        while self._server.hasPendingConnections():
            socket = self._server.nextPendingConnection()
            socket.readyRead.connect(lambda socket=socket: self._read_request(socket))
            socket.disconnected.connect(socket.deleteLater)
            if socket.bytesAvailable():
                self._read_request(socket)

    def _read_request(self, socket: QLocalSocket) -> None:
        request = bytes(socket.readAll())
        if self._REQUEST.strip() not in request.strip().splitlines():
            socket.disconnectFromServer()
            return
        socket.write(self._REPLY)
        socket.flush()
        if self._activate is not None:
            self._activate()
        socket.disconnectFromServer()
