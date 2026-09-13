"""Create a deterministic synthetic photo fixture for release evidence.

The fixture is generated from pixels only; it never copies or examines private
photos. Its manifest records checksums that release checks verify before use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw


FIXTURE_VERSION = "1"
DEFAULT_SEED = 20260913
DEFAULT_PHOTO_COUNT = 128
RECOGNIZED_OUTPUT_NAMES = frozenset({"release_fixture", "clusterlens_release_fixture"})


def build_fixture(
    output_dir: Path,
    *,
    photo_count: int = DEFAULT_PHOTO_COUNT,
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
) -> dict[str, object]:
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"fixture output is not empty: {output_dir}. Pass --overwrite to replace it.")
        if output_dir.name not in RECOGNIZED_OUTPUT_NAMES:
            raise ValueError(
                "--overwrite only accepts a directory named release_fixture or clusterlens_release_fixture."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    photo_root = output_dir / "photos"
    photo_root.mkdir()

    aggregate = hashlib.sha256()
    count = max(0, int(photo_count))
    for index in range(count):
        relative_path = Path(f"group_{index % 8:02d}") / f"synthetic_{index:06d}.png"
        image_path = photo_root / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        _write_synthetic_photo(image_path, index=index, seed=int(seed))
        aggregate.update(relative_path.as_posix().encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(_sha256_file(image_path).encode("ascii"))
        aggregate.update(b"\n")

    manifest: dict[str, object] = {
        "fixture_version": FIXTURE_VERSION,
        "generator": "scripts/create_release_fixture.py",
        "private_source_photos": False,
        "seed": int(seed),
        "photos": {
            "count": count,
            "root": "photos",
            "aggregate_sha256": aggregate.hexdigest(),
        },
    }
    manifest["manifest_sha256"] = _manifest_digest(manifest)
    (output_dir / "fixture_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _write_synthetic_photo(path: Path, *, index: int, seed: int) -> None:
    width, height = 160, 120
    base = (seed + index * 7919) & 0xFFFFFF
    background = ((base >> 16) & 0xFF, (base >> 8) & 0xFF, base & 0xFF)
    accent = ((base >> 3) & 0xFF, (base >> 11) & 0xFF, (base >> 19) & 0xFF)
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    for offset in range(-height, width, 16):
        draw.line((offset, 0, offset + height, height), fill=accent, width=3)
    draw.rectangle((12, 12, width - 13, height - 13), outline=(255, 255, 255), width=2)
    image.save(path, format="PNG", compress_level=9, optimize=False)


def _manifest_digest(manifest: dict[str, object]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--photos", type=int, default=DEFAULT_PHOTO_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true", help="Replace only a recognized fixture output directory.")
    args = parser.parse_args(argv)
    print(json.dumps(build_fixture(args.output_dir, photo_count=args.photos, seed=args.seed, overwrite=args.overwrite), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
