from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


REPORT_VERSION = "1"


@dataclass(frozen=True)
class BenchmarkStage:
    name: str
    elapsed_s: float
    details: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "elapsed_s": round(float(self.elapsed_s), 6),
        }
        if self.details:
            payload["details"] = dict(self.details)
        return payload


def build_report(
    *,
    app_id: str,
    variant: str,
    scenario: str,
    runtime_root: str,
    stages: list[BenchmarkStage],
    summary: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    total_stage_time_s = round(sum(stage.elapsed_s for stage in stages), 6)
    return {
        "report_version": REPORT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "app_id": app_id,
        "variant": variant,
        "scenario": scenario,
        "runtime_root": runtime_root,
        "total_stage_time_s": total_stage_time_s,
        "stages": [stage.as_dict() for stage in stages],
        "summary": dict(summary or {}),
        "metadata": dict(metadata or {}),
    }


def write_report(report: dict[str, object], report_dir: str | Path, stem: str) -> tuple[Path, Path]:
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_report_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def _report_markdown(report: dict[str, object]) -> str:
    stages = list(report.get("stages") or [])
    summary = dict(report.get("summary") or {})
    metadata = dict(report.get("metadata") or {})
    sorted_stages = sorted(
        (
            {
                "name": str(stage.get("name", "")),
                "elapsed_s": float(stage.get("elapsed_s", 0.0) or 0.0),
                "details": dict(stage.get("details") or {}),
            }
            for stage in stages
        ),
        key=lambda item: item["elapsed_s"],
        reverse=True,
    )
    lines = [
        f"# Benchmark Report: {report.get('app_id', 'unknown')}",
        "",
        f"- Variant: `{report.get('variant', '')}`",
        f"- Scenario: `{report.get('scenario', '')}`",
        f"- Created: `{report.get('created_at_utc', '')}`",
        f"- Runtime root: `{report.get('runtime_root', '')}`",
        f"- Total stage time: `{report.get('total_stage_time_s', 0.0)}` s",
        "",
        "## Hotspots",
    ]
    for stage in sorted_stages:
        lines.append(f"- `{stage['name']}`: `{round(stage['elapsed_s'], 6)}` s")
    if summary:
        lines.extend(["", "## Summary"])
        for key, value in sorted(summary.items()):
            lines.append(f"- `{key}`: `{_format_markdown_value(value)}`")
    if metadata:
        lines.extend(["", "## Metadata"])
        for key, value in sorted(metadata.items()):
            lines.append(f"- `{key}`: `{_format_markdown_value(value)}`")
    if sorted_stages:
        lines.extend(["", "## Stage Details"])
        for stage in sorted_stages:
            lines.append(f"### {stage['name']}")
            lines.append(f"- Elapsed: `{round(stage['elapsed_s'], 6)}` s")
            for key, value in sorted(stage["details"].items()):
                lines.append(f"- `{key}`: `{_format_markdown_value(value)}`")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _format_markdown_value(value: object) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, indent=2, sort_keys=True)
    return str(value)
