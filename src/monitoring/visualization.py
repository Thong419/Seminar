"""Lightweight PNG chart generation for monitoring reports."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:  # pragma: no cover - font availability differs across hosts
        return ImageFont.load_default()


def _base_canvas(width: int = 1200, height: int = 700) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), "white")
    return image, ImageDraw.Draw(image)


def _save(image: Image.Image, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
    return output_path


def save_line_chart_png(values: Sequence[float], output_path: Path, title: str, y_label: str) -> Path:
    image, draw = _base_canvas()
    width, height = image.size
    left, top, right, bottom = 100, 90, width - 80, height - 100
    draw.text((left, 30), title, fill="#111827", font=_font(28))
    draw.line((left, bottom, right, bottom), fill="#94a3b8", width=2)
    draw.line((left, top, left, bottom), fill="#94a3b8", width=2)

    if not values:
        draw.text((left + 20, top + 20), "No data available", fill="#6b7280", font=_font(20))
        return _save(image, output_path)

    max_value = max(values) or 1.0
    min_value = min(values)
    span = max(max_value - min_value, 1e-9)
    step_x = (right - left) / max(len(values) - 1, 1)
    points: list[tuple[float, float]] = []
    for index, value in enumerate(values):
        x = left + index * step_x
        y = bottom - ((value - min_value) / span) * (bottom - top)
        points.append((x, y))

    for start, end in zip(points, points[1:]):
        draw.line((*start, *end), fill="#2563eb", width=4)
    for x, y in points:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#1d4ed8")

    draw.text((left, bottom + 18), y_label, fill="#475569", font=_font(18))
    return _save(image, output_path)


def save_bar_chart_png(labels: Sequence[str], values: Sequence[float], output_path: Path, title: str) -> Path:
    image, draw = _base_canvas()
    width, height = image.size
    left, top, right, bottom = 100, 100, width - 80, height - 120
    draw.text((left, 30), title, fill="#111827", font=_font(28))

    if not labels or not values:
        draw.text((left + 20, top + 20), "No data available", fill="#6b7280", font=_font(20))
        return _save(image, output_path)

    max_value = max(values) or 1.0
    bar_width = max(40, (right - left) / max(len(values) * 2, 1))
    gap = bar_width
    x = left
    for label, value in zip(labels, values):
        bar_height = ((value / max_value) * (bottom - top)) if max_value else 0.0
        draw.rectangle((x, bottom - bar_height, x + bar_width, bottom), fill="#0f766e")
        draw.text((x, bottom + 8), label[:14], fill="#475569", font=_font(16))
        draw.text((x, bottom - bar_height - 22), f"{value:.2f}", fill="#111827", font=_font(16))
        x += bar_width + gap

    return _save(image, output_path)
