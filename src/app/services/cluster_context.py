from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image

from infra.cancel import raise_if_cancelled
from infra.logging_config import get_logger

from .library_catalog import CatalogAsset, ClusterContextRecord, LibraryCatalogService


LOGGER = get_logger(__name__)
CLUSTER_CONTEXT_PROMPT_VERSION = "cluster-context-v1"
MAX_CONTEXT_METADATA_CHARS = 24000
MAX_CONTEXT_IMAGE_EDGE = 1600


class ClusterContextError(RuntimeError):
    pass


@dataclass(frozen=True)
class VisionLanguageSettings:
    provider: str = "ollama"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = ""
    openai_url: str = ""
    openai_model: str = ""
    api_key_environment: str = "CLUSTERLENS_LLM_API_KEY"
    remote_consent: bool = False

    @property
    def model(self) -> str:
        return self.ollama_model.strip() if self.provider == "ollama" else self.openai_model.strip()

    @property
    def is_remote(self) -> bool:
        return self.provider == "openai-compatible"


@dataclass(frozen=True)
class VisionLanguageResponse:
    title: str
    description: str
    keywords: tuple[str, ...]
    raw_response: str = ""


class VisionLanguageProvider(Protocol):
    provider_id: str

    def describe(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        prompt: str,
        settings: VisionLanguageSettings,
        cancel_check: Callable[[], bool] | None = None,
    ) -> VisionLanguageResponse:
        ...


class OllamaVisionProvider:
    provider_id = "ollama"

    def describe(self, *, image_bytes: bytes, mime_type: str, prompt: str, settings: VisionLanguageSettings, cancel_check=None) -> VisionLanguageResponse:
        del mime_type
        _raise_cancelled(cancel_check)
        model = settings.model
        if not model:
            raise ClusterContextError("Choose a local Ollama vision model before generating cluster context.")
        payload = {
            "model": model,
            "stream": False,
            "format": "json",
            "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(image_bytes).decode("ascii")]}],
        }
        endpoint = f"{settings.ollama_url.rstrip('/')}/api/chat"
        response = _post_json(endpoint, payload, headers={}, cancel_check=cancel_check)
        message = dict(response.get("message", {}) or {})
        raw = str(message.get("content", "") or "")
        return _response_from_text(raw)


class OpenAICompatibleVisionProvider:
    provider_id = "openai-compatible"

    def describe(self, *, image_bytes: bytes, mime_type: str, prompt: str, settings: VisionLanguageSettings, cancel_check=None) -> VisionLanguageResponse:
        _raise_cancelled(cancel_check)
        if not settings.remote_consent:
            raise ClusterContextError("Remote vision generation requires explicit consent for this run.")
        model = settings.model
        endpoint = settings.openai_url.rstrip("/")
        if not model or not endpoint:
            raise ClusterContextError("Configure an OpenAI-compatible endpoint and vision model before generating cluster context.")
        api_key = os.environ.get(settings.api_key_environment, "").strip()
        if not api_key:
            raise ClusterContextError(f"Set {settings.api_key_environment} before using the remote vision provider.")
        data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
        payload = {
            "model": model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]}],
        }
        response = _post_json(endpoint, payload, headers={"Authorization": f"Bearer {api_key}"}, cancel_check=cancel_check)
        choices = list(response.get("choices", []) or [])
        if not choices:
            raise ClusterContextError("Remote vision provider returned no completion.")
        raw = str(dict(choices[0]).get("message", {}).get("content", "") or "")
        return _response_from_text(raw)


class ClusterContextService:
    """Describe one deterministic representative image for a cluster.

    The service has no source-write API. It deliberately stores only the
    generated context in the catalog, allowing the user to clear it as a
    rebuildable cache category.
    """

    def __init__(self, catalog: LibraryCatalogService | None = None) -> None:
        self.catalog = catalog or LibraryCatalogService()
        self.providers: dict[str, VisionLanguageProvider] = {
            "ollama": OllamaVisionProvider(),
            "openai-compatible": OpenAICompatibleVisionProvider(),
        }

    def describe_cluster(
        self,
        *,
        cluster_key: str,
        members: list[str] | tuple[str, ...],
        settings: VisionLanguageSettings,
        force: bool = False,
        progress_callback: Callable[[int, str], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> ClusterContextRecord:
        normalized_members = tuple(dict.fromkeys(self.catalog.canonical_path(path) for path in members if str(path or "").strip()))
        if not normalized_members:
            raise ClusterContextError("Select a non-empty cluster before generating context.")
        provider = self.providers.get(settings.provider)
        if provider is None:
            raise ClusterContextError(f"Unsupported vision provider: {settings.provider}")
        if settings.is_remote and not settings.remote_consent:
            raise ClusterContextError("Remote vision generation requires explicit consent for this run.")
        if progress_callback:
            progress_callback(0, "Preparing representative photo and metadata")
        _raise_cancelled(cancel_check)
        representative = self.catalog.representative_for_members(normalized_members)
        if not representative or not Path(representative).is_file():
            raise ClusterContextError("The representative photo is no longer available.")
        # Always read current scalar EXIF/XMP and the source revision. A
        # catalog scan may be older than a just-edited representative photo;
        # using its cached metadata here would wrongly reuse a prior context.
        # This remains one worker-side, source-read-only metadata read.
        asset = self.catalog.read_asset_metadata(representative)
        metadata = self._metadata_payload(asset)
        metadata_fingerprint = _fingerprint(
            {
                "metadata": metadata,
                "source_revision": [
                    int(asset.mtime_ns),
                    int(asset.file_size),
                    int(asset.metadata_mtime_ns),
                    int(asset.metadata_size),
                ],
            }
        )
        cache_key = self.catalog.build_context_cache_key(
            cluster_key=str(cluster_key), members=normalized_members, representative_path=representative,
            metadata_fingerprint=metadata_fingerprint, provider=provider.provider_id, model=settings.model,
            prompt_version=CLUSTER_CONTEXT_PROMPT_VERSION,
        )
        if not force:
            cached = self.catalog.load_cluster_context(cache_key)
            if cached is not None and cached.status == "ok":
                if progress_callback:
                    progress_callback(100, "Using cached cluster context")
                return cached
        if progress_callback:
            progress_callback(18, "Preparing vision input")
        image_bytes, mime_type = _read_representative_image(representative, cancel_check=cancel_check)
        _raise_cancelled(cancel_check)
        if progress_callback:
            progress_callback(35, f"Describing representative with {provider.provider_id}")
        response = provider.describe(
            image_bytes=image_bytes,
            mime_type=mime_type,
            prompt=_prompt_for_metadata(metadata),
            settings=settings,
            cancel_check=cancel_check,
        )
        _raise_cancelled(cancel_check)
        if progress_callback:
            progress_callback(82, "Saving cluster context")
        record = self.catalog.store_cluster_context(
            cluster_key=str(cluster_key), cache_key=cache_key, representative_path=representative,
            title=response.title, description=response.description, keywords=response.keywords,
            provider=provider.provider_id, model=settings.model, status="ok", members=normalized_members,
        )
        if progress_callback:
            progress_callback(100, "Cluster context ready")
        return record

    @staticmethod
    def _metadata_payload(asset: CatalogAsset) -> dict[str, object]:
        # EXIF/XMP are requested by the user. Keep scalar textual values, while
        # excluding binary maker-note/thumbnail payloads from model context.
        exif = {str(key): str(value) for key, value in asset.exif.items() if str(value).strip()}
        return {
            "image_path": asset.image_path,
            "captured_at": asset.captured_at,
            "capture_source": asset.capture_source,
            "camera": asset.camera,
            "dimensions": [int(asset.width), int(asset.height)],
            "file_extension": asset.file_ext,
            "exif": exif,
            "xmp": str(asset.xmp_text or "")[:MAX_CONTEXT_METADATA_CHARS],
        }


def _read_representative_image(image_path: str, *, cancel_check=None) -> tuple[bytes, str]:
    _raise_cancelled(cancel_check)
    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image.thumbnail((MAX_CONTEXT_IMAGE_EDGE, MAX_CONTEXT_IMAGE_EDGE), Image.Resampling.LANCZOS)
            output = BytesIO()
            image.save(output, format="JPEG", quality=88, optimize=True)
    except Exception as exc:
        raise ClusterContextError(f"Could not prepare representative photo: {exc}") from exc
    _raise_cancelled(cancel_check)
    return output.getvalue(), "image/jpeg"


def _prompt_for_metadata(metadata: dict[str, object]) -> str:
    rendered_metadata = json.dumps(metadata, ensure_ascii=False, sort_keys=True)[:MAX_CONTEXT_METADATA_CHARS]
    return (
        "Describe the visual scene represented by this one photo for a local photo-archive cluster. "
        "Use only visible evidence and the supplied metadata. Do not infer names, identities, dates, locations, "
        "or events that are not explicit. Return JSON only with title (short), description (one or two factual "
        "sentences), and keywords (up to eight concise strings).\n\n"
        f"Representative metadata:\n{rendered_metadata}"
    )


def _response_from_text(raw: str) -> VisionLanguageResponse:
    text = str(raw or "").strip()
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    title = " ".join(str(payload.get("title", "") or "").split())[:160]
    description = " ".join(str(payload.get("description", "") or "").split())[:1600]
    raw_keywords = payload.get("keywords", [])
    if isinstance(raw_keywords, str):
        raw_keywords = raw_keywords.split(",")
    keywords = tuple(
        dict.fromkeys(" ".join(str(item or "").split())[:96] for item in list(raw_keywords or []) if str(item or "").strip())
    )[:8]
    if not title and description:
        title = description.split(".", 1)[0][:160]
    if not title or not description:
        raise ClusterContextError("Vision provider returned an incomplete description. Expected JSON title and description.")
    return VisionLanguageResponse(title=title, description=description, keywords=keywords, raw_response=text[:8192])


def _post_json(endpoint: str, payload: dict[str, object], *, headers: dict[str, str], cancel_check=None) -> dict[str, object]:
    _raise_cancelled(cancel_check)
    body = json.dumps(payload).encode("utf-8")
    request = Request(endpoint, data=body, headers={"Content-Type": "application/json", **headers}, method="POST")
    try:
        with urlopen(request, timeout=120) as response:  # noqa: S310 - user-configured local/explicit endpoint
            raw = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        raise ClusterContextError(f"Vision provider request failed: HTTP {exc.code}") from exc
    except URLError as exc:
        raise ClusterContextError(f"Vision provider is unavailable: {exc.reason}") from exc
    except OSError as exc:
        raise ClusterContextError(f"Vision provider request failed: {exc}") from exc
    _raise_cancelled(cancel_check)
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ClusterContextError("Vision provider returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise ClusterContextError("Vision provider returned an unexpected response.")
    return payload


def _raise_cancelled(cancel_check: Callable[[], bool] | None) -> None:
    raise_if_cancelled(cancel_check)


def _fingerprint(payload: object) -> str:
    return __import__("hashlib").sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
