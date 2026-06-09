from __future__ import annotations


SUPPORTED_SIMILARITY_MODES = ("semantic", "cosine")
SIMILARITY_SPACE_VERSION = "semantic-pca50-cosine-full-v1"


def normalize_similarity_modes(
    similarity_modes: list[str] | tuple[str, ...] | None = None,
    legacy_similarity_mode: str | None = None,
) -> list[str]:
    raw_modes: list[str] = []
    if similarity_modes:
        raw_modes.extend(str(mode) for mode in similarity_modes)
    elif legacy_similarity_mode:
        raw_modes.append(str(legacy_similarity_mode))
    else:
        raw_modes.append("semantic")

    normalized: list[str] = []
    for mode in raw_modes:
        cleaned = str(mode or "").strip().lower()
        if cleaned not in SUPPORTED_SIMILARITY_MODES:
            cleaned = "semantic"
        if cleaned not in normalized:
            normalized.append(cleaned)
    return normalized or ["semantic"]
