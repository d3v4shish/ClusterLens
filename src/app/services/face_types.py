from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EditableFaceInput:
    """Dependency-light input contract for manually edited face boxes."""

    bbox: tuple[int, int, int, int]
    confidence: float = 1.0
