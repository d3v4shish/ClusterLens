from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from infra.cancel import raise_if_cancelled

from .library_catalog import CatalogAsset, CatalogQuery, LibraryCatalogService
from .perceptual_hash import PerceptualHashIndexService
from .similarity_search import GlobalImageIndexService


@dataclass(frozen=True)
class DuplicateCandidate:
    image_path: str
    captured_at: str
    width: int
    height: int
    file_size: int
    hash_distance: int = -1
    visual_score: float | None = None


@dataclass(frozen=True)
class DuplicateGroup:
    group_id: str
    kind: str
    keeper_path: str
    members: tuple[DuplicateCandidate, ...]
    summary: str
    review_keys: tuple[str, ...]


class DuplicateReviewService:
    """Build deterministic review-only exact, near, and burst groups."""

    def __init__(
        self,
        *,
        catalog: LibraryCatalogService | None = None,
        phash_service: PerceptualHashIndexService | None = None,
        image_index: GlobalImageIndexService | None = None,
    ) -> None:
        self.catalog = catalog or LibraryCatalogService()
        self.phash_service = phash_service or PerceptualHashIndexService()
        self.image_index = image_index or GlobalImageIndexService()

    def build_groups(
        self,
        *,
        root_ids: Iterable[str] = (),
        scope_paths: Iterable[str] | None = None,
        embedding_model: str = "siglip",
        near_hash_distance: int = 6,
        near_visual_score: float = 0.92,
        burst_seconds: int = 3,
        progress_callback: Callable[[int, str], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> list[DuplicateGroup]:
        assets = self._all_assets(root_ids, scope_paths=scope_paths)
        if not assets:
            return []
        paths = [asset.image_path for asset in assets]
        by_path = {asset.image_path: asset for asset in assets}
        if progress_callback:
            progress_callback(0, "Preparing duplicate fingerprints")
        exact_pairs = self._exact_pairs(assets, progress_callback=progress_callback, cancel_check=cancel_check)
        if progress_callback:
            progress_callback(35, "Loading perceptual hashes")
        hashes = self.phash_service.load_hash_values(paths, cancel_check=cancel_check)
        vectors = self._vectors_for_paths(paths, embedding_model, cancel_check=cancel_check)
        if progress_callback:
            progress_callback(55, "Finding near-duplicate candidates")
        near_pairs = self._near_pairs(
            hashes, vectors, max_hash_distance=max(0, int(near_hash_distance)), min_visual_score=float(near_visual_score),
            exact_pairs=exact_pairs, cancel_check=cancel_check,
        )
        if progress_callback:
            progress_callback(75, "Finding burst candidates")
        burst_pairs = self._burst_pairs(
            assets, hashes, vectors, seconds=max(0, int(burst_seconds)), cancel_check=cancel_check,
        )
        groups: list[DuplicateGroup] = []
        for kind, pairs in (("exact", exact_pairs), ("near", near_pairs), ("burst", burst_pairs)):
            groups.extend(self._groups_for_pairs(kind, pairs, by_path))
        suppressed = self.catalog.duplicate_feedback(key for group in groups for key in group.review_keys)
        filtered = [group for group in groups if not all(suppressed.get(key) == "not-duplicate" for key in group.review_keys)]
        filtered.sort(key=lambda group: ({"exact": 0, "near": 1, "burst": 2}.get(group.kind, 9), -len(group.members), group.keeper_path))
        if progress_callback:
            progress_callback(100, f"Prepared {len(filtered)} duplicate and burst review groups")
        return filtered

    def mark_not_duplicate(
        self,
        group: DuplicateGroup,
        *,
        progress_callback: Callable[[int, str], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        keeper = group.keeper_path
        review_keys = iter(group.review_keys)
        candidates = tuple(candidate for candidate in group.members if candidate.image_path != keeper)
        for index, candidate in enumerate(candidates, start=1):
            raise_if_cancelled(cancel_check)
            if candidate.image_path == keeper:
                continue
            key = next(review_keys, "")
            if not key:
                continue
            self.catalog.set_duplicate_feedback(decision_key=key, kind=group.kind, left_path=keeper, right_path=candidate.image_path, state="not-duplicate")
            if progress_callback:
                progress_callback(int(index * 100 / max(1, len(candidates))), f"Saving duplicate decision {index}/{len(candidates)}")

    @staticmethod
    def pair_key(
        kind: str,
        left_path: str,
        right_path: str,
        hash_distance: int,
        visual_score: float | None,
        *,
        left_fingerprint: str = "",
        right_fingerprint: str = "",
    ) -> str:
        paths = sorted((str(left_path), str(right_path)))
        fingerprints = sorted((str(left_fingerprint), str(right_fingerprint)))
        payload = (
            f"{kind}|{paths[0]}|{paths[1]}|{fingerprints[0]}|{fingerprints[1]}|"
            f"{int(hash_distance)}|{'' if visual_score is None else round(float(visual_score), 6)}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _all_assets(self, root_ids: Iterable[str], *, scope_paths: Iterable[str] | None = None) -> list[CatalogAsset]:
        root_ids = tuple(str(item) for item in root_ids if str(item))
        roots_scope = tuple(str(item) for item in scope_paths if str(item)) if scope_paths is not None else None
        offset = 0
        assets: list[CatalogAsset] = []
        while True:
            page = self.catalog.query_assets(CatalogQuery(root_ids=root_ids, scope_paths=roots_scope, order="captured_asc", offset=offset, limit=1000))
            assets.extend(page.items)
            if page.next_offset is None:
                return assets
            offset = page.next_offset

    @staticmethod
    def _exact_pairs(assets: list[CatalogAsset], *, progress_callback=None, cancel_check=None) -> list[tuple[str, str, int, float | None]]:
        by_size: dict[int, list[CatalogAsset]] = defaultdict(list)
        for asset in assets:
            by_size[int(asset.file_size)].append(asset)
        candidates = [asset for group in by_size.values() if len(group) > 1 for asset in group]
        digests: dict[str, list[str]] = defaultdict(list)
        for index, asset in enumerate(candidates, start=1):
            raise_if_cancelled(cancel_check)
            try:
                digest = _sha256_file(asset.image_path, cancel_check=cancel_check)
            except OSError:
                continue
            digests[digest].append(asset.image_path)
            if progress_callback and (index == len(candidates) or index % 24 == 0):
                progress_callback(int(index * 30 / max(1, len(candidates))), f"Hashing exact candidates {index}/{len(candidates)}")
        pairs: list[tuple[str, str, int, float | None]] = []
        for paths in digests.values():
            ordered = sorted(paths)
            if len(ordered) < 2:
                continue
            first = ordered[0]
            pairs.extend((first, other, 0, 1.0) for other in ordered[1:])
        return pairs

    def _vectors_for_paths(self, paths: list[str], model: str, *, cancel_check=None) -> dict[str, np.ndarray]:
        try:
            records = self.image_index.load_records_for_paths(model, paths, cancel_check=cancel_check)
        except Exception:
            return {}
        wanted = set(paths)
        return {record.image_path: np.asarray(record.embedding, dtype=np.float32) for record in records if record.image_path in wanted}

    def _near_pairs(self, hashes, vectors, *, max_hash_distance: int, min_visual_score: float, exact_pairs, cancel_check=None) -> list[tuple[str, str, int, float | None]]:
        exact_path_pairs = {tuple(sorted((left, right))) for left, right, _distance, _score in exact_pairs}
        buckets: dict[tuple[int, int], list[str]] = defaultdict(list)
        for path, value in hashes.items():
            for band in range(4):
                buckets[(band, (int(value) >> (band * 16)) & 0xFFFF)].append(path)
        pairs: dict[tuple[str, str], tuple[int, float | None]] = {}
        for paths in buckets.values():
            ordered = sorted(set(paths))
            if len(ordered) < 2:
                continue
            for left_index, left in enumerate(ordered[:-1]):
                for right in ordered[left_index + 1 :]:
                    raise_if_cancelled(cancel_check)
                    key = (left, right)
                    if key in exact_path_pairs or key in pairs:
                        continue
                    distance = int((int(hashes[left]) ^ int(hashes[right])).bit_count())
                    if distance > max_hash_distance:
                        continue
                    score = _cosine(vectors.get(left), vectors.get(right))
                    # An unavailable visual vector means the candidate remains
                    # reviewable but is never considered an auto action.
                    if score is not None and score < min_visual_score:
                        continue
                    pairs[key] = (distance, score)
        return [(left, right, distance, score) for (left, right), (distance, score) in pairs.items()]

    def _burst_pairs(self, assets, hashes, vectors, *, seconds: int, cancel_check=None) -> list[tuple[str, str, int, float | None]]:
        ordered = sorted(assets, key=lambda item: (_timestamp(item.captured_at), item.image_path))
        pairs: list[tuple[str, str, int, float | None]] = []
        for left_index, left in enumerate(ordered):
            left_time = _timestamp(left.captured_at)
            for right in ordered[left_index + 1 :]:
                raise_if_cancelled(cancel_check)
                right_time = _timestamp(right.captured_at)
                if right_time - left_time > seconds:
                    break
                if Path(left.image_path).parent != Path(right.image_path).parent:
                    continue
                hash_distance = -1
                if left.image_path in hashes and right.image_path in hashes:
                    hash_distance = int((int(hashes[left.image_path]) ^ int(hashes[right.image_path])).bit_count())
                score = _cosine(vectors.get(left.image_path), vectors.get(right.image_path))
                if score is None:
                    # A time-neighbour alone is not a useful burst suggestion.
                    continue
                if score >= 0.80:
                    pairs.append((left.image_path, right.image_path, hash_distance, score))
        return pairs

    def _groups_for_pairs(self, kind: str, pairs, assets_by_path: dict[str, CatalogAsset]) -> list[DuplicateGroup]:
        parent: dict[str, str] = {}
        pair_data: dict[tuple[str, str], tuple[int, float | None]] = {}
        for left, right, distance, score in pairs:
            parent.setdefault(left, left)
            parent.setdefault(right, right)
            _union(parent, left, right)
            pair_data[tuple(sorted((left, right)))] = (int(distance), score)
        components: dict[str, list[str]] = defaultdict(list)
        for path in parent:
            components[_find(parent, path)].append(path)
        groups: list[DuplicateGroup] = []
        for paths in components.values():
            ordered_paths = sorted(paths)
            assets = [assets_by_path[path] for path in ordered_paths if path in assets_by_path]
            if len(assets) < 2:
                continue
            keeper = _choose_keeper(assets)
            candidates: list[DuplicateCandidate] = []
            review_keys: list[str] = []
            for asset in sorted(assets, key=lambda item: (item.image_path != keeper.image_path, item.image_path)):
                if asset.image_path == keeper.image_path:
                    distance, score = 0, 1.0
                else:
                    distance, score = pair_data.get(tuple(sorted((keeper.image_path, asset.image_path))), (-1, None))
                    review_keys.append(
                        self.pair_key(
                            kind,
                            keeper.image_path,
                            asset.image_path,
                            distance,
                            score,
                            left_fingerprint=_asset_fingerprint(keeper),
                            right_fingerprint=_asset_fingerprint(asset),
                        )
                    )
                candidates.append(DuplicateCandidate(asset.image_path, asset.captured_at, asset.width, asset.height, asset.file_size, distance, score))
            group_id = hashlib.sha256(f"{kind}|{'|'.join(ordered_paths)}".encode("utf-8")).hexdigest()[:20]
            summary = {"exact": "Identical file bytes", "near": "Perceptual hash with optional visual verification", "burst": "Capture-time and visual-similarity burst"}.get(kind, kind)
            groups.append(DuplicateGroup(group_id, kind, keeper.image_path, tuple(candidates), summary, tuple(review_keys)))
        return groups


def _sha256_file(path: str, *, cancel_check=None) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            raise_if_cancelled(cancel_check)
            block = handle.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _cosine(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    if left is None or right is None or left.shape != right.shape or not left.size:
        return None
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return None
    return float(np.dot(left, right) / (left_norm * right_norm))


def _timestamp(value: str) -> float:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return float("inf")


def _choose_keeper(assets: list[CatalogAsset]) -> CatalogAsset:
    return sorted(assets, key=lambda item: (-int(item.width) * int(item.height), -int(item.file_size), item.captured_at, item.image_path))[0]


def _asset_fingerprint(asset: CatalogAsset) -> str:
    """Include the catalogued source revision in duplicate-review feedback."""
    return f"{int(asset.mtime_ns)}:{int(asset.file_size)}"


def _find(parent: dict[str, str], value: str) -> str:
    root = parent[value]
    while root != parent[root]:
        root = parent[root]
    while value != root:
        next_value = parent[value]
        parent[value] = root
        value = next_value
    return root


def _union(parent: dict[str, str], left: str, right: str) -> None:
    left_root = _find(parent, left)
    right_root = _find(parent, right)
    if left_root != right_root:
        parent[max(left_root, right_root)] = min(left_root, right_root)
