"""Markdown report helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def write_markdown_report(path: Path, title: str, sections: dict[str, str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}_",
        "",
    ]
    for heading, body in sections.items():
        lines.extend([f"## {heading}", "", body.strip(), ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def bullet_list(items: list[str]) -> str:
    if not items:
        return "_None._"
    return "\n".join(f"- {item}" for item in items)
