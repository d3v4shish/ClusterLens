from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SelectionTarget:
    paths: tuple[str, ...]
    kind: str
    label: str
    source_context: dict[str, object] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.paths)

    def as_list(self) -> list[str]:
        return list(self.paths)

    def contains(self, image_path: str) -> bool:
        return str(image_path) in self.paths
