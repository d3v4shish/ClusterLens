from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from infra.logging_config import get_logger
from infra.settings import get_settings


LOGGER = get_logger(__name__)
CLUSTER_MEANING_PROMPT_VERSION = "cluster-meaning-prompts-v1"
DEFAULT_CLUSTER_MEANING_MODEL = "clip"
SUPPORTED_CLUSTER_MEANING_MODELS = ("clip", "openclip", "siglip")
MAX_MEANING_IMAGES_PER_CLUSTER = 24


class ClusterMeaningCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class ClusterMeaningLabel:
    label: str
    prompt: str
    score: float

    def as_context(self) -> dict[str, object]:
        return {
            "label": str(self.label),
            "prompt": str(self.prompt),
            "score": float(self.score),
        }

    @classmethod
    def from_context(cls, context: dict[str, object]) -> "ClusterMeaningLabel":
        return cls(
            label=str(context.get("label", "") or ""),
            prompt=str(context.get("prompt", "") or ""),
            score=_maybe_float(context.get("score")) or 0.0,
        )


@dataclass(frozen=True)
class ClusterMeaning:
    cluster_id: int
    labels: tuple[ClusterMeaningLabel, ...] = ()
    confidence: str = "Unavailable"
    explanation_model: str = ""
    image_count_used: int = 0
    prompt_version: str = CLUSTER_MEANING_PROMPT_VERSION
    filename_terms: tuple[tuple[str, int], ...] = ()
    status: str = "unavailable"
    message: str = ""

    def as_context(self) -> dict[str, object]:
        return {
            "cluster_id": int(self.cluster_id),
            "labels": [label.as_context() for label in self.labels],
            "confidence": str(self.confidence),
            "explanation_model": str(self.explanation_model),
            "image_count_used": int(self.image_count_used),
            "prompt_version": str(self.prompt_version),
            "filename_terms": [[term, int(count)] for term, count in self.filename_terms],
            "status": str(self.status),
            "message": str(self.message),
        }

    @classmethod
    def from_context(cls, context: dict[str, object]) -> "ClusterMeaning":
        return cls(
            cluster_id=int(context.get("cluster_id", -1) or -1),
            labels=tuple(
                ClusterMeaningLabel.from_context(dict(label))
                for label in context.get("labels", []) or []
            ),
            confidence=str(context.get("confidence", "Unavailable") or "Unavailable"),
            explanation_model=str(context.get("explanation_model", "") or ""),
            image_count_used=int(context.get("image_count_used", 0) or 0),
            prompt_version=str(context.get("prompt_version", CLUSTER_MEANING_PROMPT_VERSION) or CLUSTER_MEANING_PROMPT_VERSION),
            filename_terms=tuple(
                (str(item[0]), int(item[1]))
                for item in context.get("filename_terms", []) or []
                if isinstance(item, (list, tuple)) and len(item) == 2
            ),
            status=str(context.get("status", "unavailable") or "unavailable"),
            message=str(context.get("message", "") or ""),
        )

    @classmethod
    def unavailable(cls, cluster_id: int, *, explanation_model: str = "", message: str = "") -> "ClusterMeaning":
        return cls(
            cluster_id=int(cluster_id),
            confidence="Unavailable",
            explanation_model=str(explanation_model),
            status="unavailable",
            message=str(message or "The app can group these images, but cannot name this cluster from model evidence."),
        )


@dataclass(frozen=True)
class PromptEntry:
    label: str
    prompt: str


PROMPT_BANK: tuple[PromptEntry, ...] = (
    PromptEntry("beach", "a photo of a beach with sand and ocean"),
    PromptEntry("ocean", "a photo of the ocean or sea"),
    PromptEntry("sunset", "a photo of a sunset or sunrise"),
    PromptEntry("mountains", "a photo of mountains or hills"),
    PromptEntry("forest", "a photo of a forest or trees"),
    PromptEntry("city", "a photo of a city street or skyline"),
    PromptEntry("architecture", "a photo of buildings or architecture"),
    PromptEntry("people", "a photo of people together"),
    PromptEntry("portrait", "a portrait photo of a person"),
    PromptEntry("wedding", "a wedding photo"),
    PromptEntry("party", "a party or celebration photo"),
    PromptEntry("food", "a photo of food or a meal"),
    PromptEntry("animals", "a photo of animals or pets"),
    PromptEntry("flowers", "a photo of flowers or plants"),
    PromptEntry("car", "a photo of a car or vehicle"),
    PromptEntry("sports", "a sports or action photo"),
    PromptEntry("snow", "a photo of snow or winter"),
    PromptEntry("night", "a night photo or low light scene"),
    PromptEntry("indoor", "an indoor room photo"),
    PromptEntry("landscape", "a landscape nature photo"),
    PromptEntry("document", "a photo of a document or paper"),
    PromptEntry("screenshot", "a screenshot of a computer or phone screen"),
    PromptEntry("product", "a product photo"),
    PromptEntry("art", "an artwork, drawing, or illustration"),
    PromptEntry("travel", "a travel vacation photo"),
)


class ClusterMeaningCacheService:
    def __init__(self, cache_dir: Path | None = None) -> None:
        settings = get_settings()
        self.cache_dir = Path(cache_dir) if cache_dir is not None else settings.cache_dir / "cluster_meanings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def build_key(
        self,
        *,
        snapshot_key: str,
        comparison_key: str,
        cluster_id: int,
        image_paths: list[str] | tuple[str, ...],
        path_fingerprints: dict[str, tuple[int, int]] | None,
        explanation_model: str,
        prompt_version: str = CLUSTER_MEANING_PROMPT_VERSION,
    ) -> str:
        fingerprint_payload = []
        for path in sorted(str(path) for path in image_paths):
            mtime_ns, size = (path_fingerprints or {}).get(path, (0, 0))
            fingerprint_payload.append([path, int(mtime_ns), int(size)])
        payload = json.dumps(
            {
                "snapshot_key": str(snapshot_key),
                "comparison_key": str(comparison_key),
                "cluster_id": int(cluster_id),
                "image_fingerprints": fingerprint_payload,
                "explanation_model": str(explanation_model),
                "prompt_version": str(prompt_version),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def load(self, key: str) -> ClusterMeaning | None:
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            return ClusterMeaning.from_context(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            LOGGER.warning("Ignoring corrupt cluster meaning cache %s: %s", path, exc)
            return None

    def save(self, key: str, meaning: ClusterMeaning) -> None:
        path = self.cache_dir / f"{key}.json"
        path.write_text(json.dumps(meaning.as_context(), indent=2), encoding="utf-8")


class ClusterMeaningService:
    def __init__(self, cache_service: ClusterMeaningCacheService | None = None) -> None:
        self.cache_service = cache_service or ClusterMeaningCacheService()
        self._prompt_embedding_cache: dict[str, list[tuple[PromptEntry, np.ndarray]]] = {}

    def resolve_model(self, requested_model: str, selected_models: list[str] | tuple[str, ...]) -> str:
        requested = str(requested_model or "auto").strip().lower()
        if requested in SUPPORTED_CLUSTER_MEANING_MODELS:
            return requested
        for model_name in selected_models:
            normalized = str(model_name or "").strip().lower()
            if normalized in SUPPORTED_CLUSTER_MEANING_MODELS:
                return normalized
        return DEFAULT_CLUSTER_MEANING_MODEL

    def generate(
        self,
        *,
        clusters_by_key: dict[str, dict[int, list[str]]],
        all_image_paths: list[str],
        snapshot_key: str,
        path_fingerprints: dict[str, tuple[int, int]] | None,
        embedding_service,
        selected_models: list[str],
        requested_model: str = "auto",
        use_embedding_cache_lookup: bool = True,
        progress_callback: Callable[[int, str], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> tuple[dict[str, dict[int, ClusterMeaning]], dict[str, object]]:
        if not clusters_by_key:
            return {}, {"cluster_meaning_status": "skipped-empty"}

        resolved_model = self.resolve_model(requested_model, selected_models)
        model_candidates = [resolved_model]
        if resolved_model == "siglip":
            model_candidates.append(DEFAULT_CLUSTER_MEANING_MODEL)

        last_error = ""
        for model_name in model_candidates:
            try:
                return self._generate_with_model(
                    clusters_by_key=clusters_by_key,
                    all_image_paths=all_image_paths,
                    snapshot_key=snapshot_key,
                    path_fingerprints=path_fingerprints,
                    embedding_service=embedding_service,
                    model_name=model_name,
                    use_embedding_cache_lookup=use_embedding_cache_lookup,
                    progress_callback=progress_callback,
                    cancel_check=cancel_check,
                )
            except Exception as exc:
                if isinstance(exc, ClusterMeaningCancelled):
                    raise
                last_error = str(exc)
                LOGGER.warning("Cluster meaning generation failed with %s: %s", model_name, exc)

        unavailable = _unavailable_for_clusters(clusters_by_key, resolved_model, last_error)
        return unavailable, {
            "cluster_meaning_status": "unavailable",
            "cluster_meaning_model": resolved_model,
            "cluster_meaning_error": last_error,
        }

    def _generate_with_model(
        self,
        *,
        clusters_by_key: dict[str, dict[int, list[str]]],
        all_image_paths: list[str],
        snapshot_key: str,
        path_fingerprints: dict[str, tuple[int, int]] | None,
        embedding_service,
        model_name: str,
        use_embedding_cache_lookup: bool,
        progress_callback: Callable[[int, str], None] | None,
        cancel_check: Callable[[], bool] | None,
    ) -> tuple[dict[str, dict[int, ClusterMeaning]], dict[str, object]]:
        started = time.perf_counter()
        if progress_callback:
            progress_callback(0, f"Generating cluster meanings with {model_name.upper()}")

        meanings: dict[str, dict[int, ClusterMeaning]] = {}
        misses: list[tuple[str, int, list[str], str]] = []
        cache_hits = 0
        cache_misses = 0

        for comparison_key, clusters in clusters_by_key.items():
            meanings[comparison_key] = {}
            for cluster_id, image_paths in clusters.items():
                if cancel_check and cancel_check():
                    raise ClusterMeaningCancelled("Cluster meaning generation cancelled.")
                cache_key = self.cache_service.build_key(
                    snapshot_key=snapshot_key,
                    comparison_key=comparison_key,
                    cluster_id=int(cluster_id),
                    image_paths=list(image_paths),
                    path_fingerprints=path_fingerprints,
                    explanation_model=model_name,
                )
                cached = self.cache_service.load(cache_key)
                if cached is not None:
                    meanings[comparison_key][int(cluster_id)] = cached
                    cache_hits += 1
                    continue
                cache_misses += 1
                misses.append((comparison_key, int(cluster_id), list(image_paths), cache_key))

        if not misses:
            return meanings, {
                "cluster_meaning_status": "cached",
                "cluster_meaning_model": model_name,
                "cluster_meaning_cache_hits": cache_hits,
                "cluster_meaning_cache_misses": cache_misses,
                "cluster_meaning_time_s": round(time.perf_counter() - started, 3),
            }

        needed_paths = _ordered_unique(
            path
            for _comparison_key, _cluster_id, image_paths, _cache_key in misses
            for path in image_paths[:MAX_MEANING_IMAGES_PER_CLUSTER]
        )
        path_fingerprints = path_fingerprints or {}

        def _embedding_progress(done: int, total: int, stage: str) -> None:
            if progress_callback:
                progress_callback(int(45 * done / max(1, total)), f"Generating cluster meanings: {stage}")

        ordered_embeddings, embed_metrics = embedding_service.embed_paths(
            needed_paths or all_image_paths,
            model_name,
            use_onnx=False,
            progress_callback=_embedding_progress,
            use_cache_lookup=use_embedding_cache_lookup,
            path_fingerprints=path_fingerprints,
        )
        embeddings_by_path = {str(path): _normalize_vector(vector) for path, vector in ordered_embeddings}
        prompt_embeddings = self._prompt_embeddings(embedding_service, model_name)

        total_misses = max(1, len(misses))
        for index, (comparison_key, cluster_id, image_paths, cache_key) in enumerate(misses, start=1):
            if cancel_check and cancel_check():
                raise ClusterMeaningCancelled("Cluster meaning generation cancelled.")
            meaning = self._meaning_for_cluster(
                cluster_id=cluster_id,
                image_paths=image_paths,
                embeddings_by_path=embeddings_by_path,
                prompt_embeddings=prompt_embeddings,
                model_name=model_name,
            )
            meanings[comparison_key][cluster_id] = meaning
            if meaning.status == "ok":
                try:
                    self.cache_service.save(cache_key, meaning)
                except OSError as exc:
                    LOGGER.warning("Could not write cluster meaning cache: %s", exc)
            if progress_callback:
                progress_callback(
                    45 + int(55 * index / total_misses),
                    f"Generated cluster meanings {index}/{total_misses}",
                )

        return meanings, {
            "cluster_meaning_status": "ok",
            "cluster_meaning_model": model_name,
            "cluster_meaning_prompt_version": CLUSTER_MEANING_PROMPT_VERSION,
            "cluster_meaning_cache_hits": cache_hits,
            "cluster_meaning_cache_misses": cache_misses,
            "cluster_meaning_time_s": round(time.perf_counter() - started, 3),
            "cluster_meaning_embedding_cache_hits": int(embed_metrics.get("cache_hits", 0) or 0),
            "cluster_meaning_embedding_cache_misses": int(embed_metrics.get("cache_misses", 0) or 0),
        }

    def _prompt_embeddings(self, embedding_service, model_name: str) -> list[tuple[PromptEntry, np.ndarray]]:
        cache_key = f"{model_name}:{CLUSTER_MEANING_PROMPT_VERSION}"
        cached = self._prompt_embedding_cache.get(cache_key)
        if cached is not None:
            return cached
        embed_texts = getattr(embedding_service, "embed_texts", None)
        if callable(embed_texts):
            vectors = embed_texts(
                [entry.prompt for entry in PROMPT_BANK],
                model_name=model_name,
                use_onnx=False,
            )
            prompt_embeddings = [
                (entry, _normalize_vector(vector))
                for entry, vector in zip(PROMPT_BANK, vectors)
            ]
        else:
            prompt_embeddings = [
                (entry, _normalize_vector(embedding_service.embed_text(entry.prompt, model_name=model_name, use_onnx=False)))
                for entry in PROMPT_BANK
            ]
        self._prompt_embedding_cache[cache_key] = prompt_embeddings
        return prompt_embeddings

    def _meaning_for_cluster(
        self,
        *,
        cluster_id: int,
        image_paths: list[str],
        embeddings_by_path: dict[str, np.ndarray],
        prompt_embeddings: list[tuple[PromptEntry, np.ndarray]],
        model_name: str,
    ) -> ClusterMeaning:
        selected_paths = [str(path) for path in image_paths[:MAX_MEANING_IMAGES_PER_CLUSTER]]
        vectors = [embeddings_by_path[path] for path in selected_paths if path in embeddings_by_path]
        if not vectors:
            return ClusterMeaning.unavailable(
                cluster_id,
                explanation_model=model_name,
                message="The app could not embed representative images for model-derived labels.",
            )

        centroid = _normalize_vector(np.asarray(vectors, dtype=np.float32).mean(axis=0))
        scored = [
            ClusterMeaningLabel(label=entry.label, prompt=entry.prompt, score=float(np.dot(centroid, text_vector)))
            for entry, text_vector in prompt_embeddings
        ]
        scored.sort(key=lambda label: label.score, reverse=True)
        top_labels = tuple(scored[:3])
        confidence = _confidence_for_scores(top_labels)
        return ClusterMeaning(
            cluster_id=cluster_id,
            labels=top_labels,
            confidence=confidence,
            explanation_model=model_name,
            image_count_used=len(vectors),
            prompt_version=CLUSTER_MEANING_PROMPT_VERSION,
            filename_terms=_filename_terms(image_paths),
            status="ok",
        )


def _confidence_for_scores(labels: tuple[ClusterMeaningLabel, ...]) -> str:
    if not labels:
        return "Unavailable"
    top = labels[0].score
    gap = top - labels[1].score if len(labels) > 1 else top
    if top >= 0.28 and gap >= 0.025:
        return "High"
    if top >= 0.22 or gap >= 0.015:
        return "Medium"
    return "Low"


def _filename_terms(image_paths: list[str] | tuple[str, ...]) -> tuple[tuple[str, int], ...]:
    stopwords = {
        "img",
        "image",
        "photo",
        "pic",
        "dsc",
        "screenshot",
        "copy",
        "edited",
        "final",
        "jpg",
        "jpeg",
        "png",
        "webp",
    }
    counter: Counter[str] = Counter()
    for image_path in image_paths:
        stem = Path(str(image_path)).stem.lower()
        for term in re.split(r"[^a-z0-9]+", stem):
            if len(term) < 3 or term.isdigit() or term in stopwords:
                continue
            counter[term] += 1
    return tuple(counter.most_common(5))


def _unavailable_for_clusters(
    clusters_by_key: dict[str, dict[int, list[str]]],
    explanation_model: str,
    message: str,
) -> dict[str, dict[int, ClusterMeaning]]:
    return {
        comparison_key: {
            int(cluster_id): ClusterMeaning.unavailable(
                int(cluster_id),
                explanation_model=explanation_model,
                message=message,
            )
            for cluster_id in clusters
        }
        for comparison_key, clusters in clusters_by_key.items()
    }


def _ordered_unique(values) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value)
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12:
        return array
    return array / norm


def _maybe_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None
