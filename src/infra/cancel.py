from __future__ import annotations

from collections.abc import Callable


class Cancelled(RuntimeError):
    """Raised to cooperatively cancel background work."""


def raise_if_cancelled(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check is not None and cancel_check():
        raise Cancelled()
