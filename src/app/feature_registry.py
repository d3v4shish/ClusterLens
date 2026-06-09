from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


FeaturePanelBuilder = Callable[[Any], object]
FeatureCallback = Callable[[Any], None]


@dataclass(frozen=True)
class FeatureModule:
    feature_id: str
    label: str
    enabled: bool = False
    build_panel: FeaturePanelBuilder | None = None
    attach: FeatureCallback | None = None
    detach: FeatureCallback | None = None
