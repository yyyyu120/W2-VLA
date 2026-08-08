#!/usr/bin/env python3
"""Create a compact side-by-side M2W label showcase video from debug PNGs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PHASES = [
    {
        "name": "approach",
        "start": 0,
        "end": 49,
        "color": (64, 129, 236),
        "primary": "Locate black bowl in top drawer",
        "wrist": "Search for bowl rim and reachable grasp area",
        "contact": "not in contact",
        "next": "Move gripper toward the bowl",
        "anchor": "pre-contact motion",
    },
    {
        "name": "contact",
        "start": 50,
        "end": 61,
        "color": (247, 151, 31),
        "primary": "Black bowl grasp point",
        "wrist": "Fingertips closing around the bowl rim",
        "contact": "touching",
        "next": "Close gripper on the bowl",
        "anchor": "gripper close",
    },
    {
        "name": "move",
        "start": 62,
        "end": 115,
        "color": (48, 162, 86),
        "primary": "Grasped black bowl",
        "wrist": "Bowl stability and path toward plate",
        "contact": "grasping",
        "next": "Transport bowl toward the plate",
        "anchor": "post-contact motion",
    },
    {
        "name": "align",
        "start": 116,
        "end": 124,
        "color": (156, 91, 208),
        "primary": "Plate placement site",
        "wrist": "Bowl-plate relative alignment",
        "contact": "grasping",
        "next": "Align bowl over the plate",
        "anchor": "goal alignment",
    },
    {
        "name": "release",
        "start": 125,
        "end": 130,
        "color": (237, 96, 80),
        "primary": "Final bowl-on-plate placement",
        "wrist": "Release clearance at the plate",
        "contact": "released",
        "next": "Open gripper and leave bowl on plate",
        "anchor": "gripper open",
    },
    {
        "name": "done",
        "start": 131,
        "end": 148,
        "color": (107, 116, 126),
        "primary": "Task-complete bowl placement",
        "wrist": "Verify bowl remains on the plate",
        "contact": "released",
        "next": "Retract gripper after placement",
        "anchor": "task complete",
    },
]


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_TITLE = font(34, bold=True)
FONT_SUBTITLE = font(18)
FONT_LABEL = font(18, bold=True)
FONT_BODY = font(27, bold=True)
FONT_SMALL = font(16)
FONT_TINY = font(13)


def phase_for_step(step: int) -> dict:
    for phase in PHASES:
        if phase["start"] <= step <= phase["end"]:
            return phase
    if step < PHASES[0]["start"]:
        return PHASES[0]
    return PHASES[-1]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, text_font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if draw.textlength(candidate, font=text_font) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_text_block(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    label: str,
    value: str,
    color: tuple[int, int, int],
    max_width: int,
) -> int:
    draw.text((x, y), label, fill=(108, 119, 136), font=FONT_LABEL)
    y += 28
    for line in wrap_text(draw, value, FONT_BODY, max_width):
        draw.text((x, y), line, fill=color, font=FONT_BODY)
        y += 34
    return y + 14


def draw_timeline(draw: ImageDraw.ImageDraw, x: int, y: int, width: int, step: int) -> None:
    max_step = max(phase["end"] for phase in PHASES)
    track_h = 16
    for phase in PHASES:
        sx = x + int(width * phase["start"] / max_step)
        ex = x + int(width * phase["end"] / max_step)
        draw.rounded_rectangle((sx, y, max(ex, sx + 2), y + track_h), radius=8, fill=phase["color"])
    marker_x = x + int(width * max(0, min(step, max_step)) / max_step)
    draw.line((marker_x, y - 8, marker_x, y + track_h + 8), fill=(17, 24, 39), width=4)
    draw.ellipse((marker_x - 7, y - 12, marker_x + 7, y + 2), fill=(17, 24, 39))


def resize_cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_w, target_h = size
    scale = max(target_w / image.width, target_h / image.height)
    resized = image.resize(
        (math.ceil(image.width * scale), math.ceil(image.height * scale)),
        Image.Resampling.LANCZOS,
    )
    left = (resized.width - target_w) // 2
    top = (resized.height - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def crop_debug_image(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    w, h = image.size
    # The existing debug image is header + square render + verbose text panel.
    # The small "single" camera tag sits just under the header, so trim it too.
    render_top = 22 if h >= 460 else 0
    top = min(render_top + 18, h)
    bottom = min(h, render_top + w)
    return image.crop((0, top, w, max(top + 1, bottom)))


def render_frame(path: Path, out_size: tuple[int, int]) -> np.ndarray:
    step = int(path.stem.split("_")[-1])
    phase = phase_for_step(step)
    phase_color = phase["color"]

    canvas_w, canvas_h = out_size
    left_w = 720
    right_x = left_w
    panel_w = canvas_w - left_w

    canvas = Image.new("RGB", out_size, (246, 248, 251))
    draw = ImageDraw.Draw(canvas)

    visual = resize_cover(crop_debug_image(path), (left_w, canvas_h))
    canvas.paste(visual, (0, 0))

    overlay = Image.new("RGBA", (left_w, 110), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rectangle((0, 0, left_w, 8), fill=phase_color + (255,))
    od.rounded_rectangle((24, 22, 210, 82), radius=12, fill=(17, 24, 39, 210))
    od.text((42, 32), f"STEP {step:03d}", fill=(255, 255, 255, 255), font=FONT_LABEL)
    od.text((42, 55), phase["name"].upper(), fill=phase_color + (255,), font=FONT_SMALL)
    canvas.paste(overlay.convert("RGB"), (0, 0), overlay)

    draw.rectangle((right_x, 0, canvas_w, canvas_h), fill=(250, 251, 253))
    draw.rectangle((right_x, 0, right_x + 7, canvas_h), fill=phase_color)
    x = right_x + 48
    y = 42

    draw.text((x, y), "W^2-VLA Temporal Grounding", fill=(18, 25, 38), font=FONT_TITLE)
    y += 44
    draw.text((x, y), "main-view plan -> wrist-view local evidence", fill=(94, 106, 126), font=FONT_SUBTITLE)
    y += 42

    badge_w = int(draw.textlength(phase["name"].upper(), font=FONT_BODY)) + 42
    draw.rounded_rectangle((x, y, x + badge_w, y + 48), radius=16, fill=phase_color)
    draw.text((x + 21, y + 8), phase["name"].upper(), fill=(255, 255, 255), font=FONT_BODY)
    draw.text((x + badge_w + 20, y + 14), f"step {step:03d} / 148", fill=(94, 106, 126), font=FONT_LABEL)
    y += 78

    max_text_w = panel_w - 96
    y = draw_text_block(draw, x, y, "PrimaryFocus", phase["primary"], (20, 29, 43), max_text_w)
    y = draw_text_block(draw, x, y, "WristFocus", phase["wrist"], (20, 29, 43), max_text_w)
    y = draw_text_block(draw, x, y, "Contact", phase["contact"], phase_color, max_text_w)
    y = draw_text_block(draw, x, y, "NextMotion", phase["next"], (20, 29, 43), max_text_w)

    draw.rounded_rectangle((x, canvas_h - 154, canvas_w - 44, canvas_h - 82), radius=14, fill=(238, 242, 247))
    draw.text((x + 22, canvas_h - 140), "Temporal Anchor", fill=(108, 119, 136), font=FONT_SMALL)
    draw.text((x + 22, canvas_h - 116), phase["anchor"], fill=(20, 29, 43), font=FONT_LABEL)
    draw_timeline(draw, x, canvas_h - 50, panel_w - 96, step)

    return np.asarray(canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument(
        "--duration",
        type=float,
        help="Target duration in seconds. Overrides --fps after frame duplication.",
    )
    parser.add_argument("--hold", type=int, default=2, help="Duplicate each sampled frame this many times.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    frame_dir = Path(args.frame_dir)
    frame_paths = sorted(frame_dir.glob("step_*.png"))
    if not frame_paths:
        raise SystemExit(f"No step_*.png files found in {frame_dir}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for path in frame_paths:
        frame = render_frame(path, (args.width, args.height))
        frames.extend([frame] * max(1, int(args.hold)))

    try:
        fps = len(frames) / args.duration if args.duration else args.fps
        imageio.mimsave(output.as_posix(), frames, fps=fps, quality=8, macro_block_size=16)
    except Exception:
        fallback = output.with_suffix(".gif")
        imageio.mimsave(fallback.as_posix(), frames, fps=args.fps)
        raise SystemExit(f"MP4 writer failed; wrote GIF fallback to {fallback}")

    duration = len(frames) / float(fps)
    print(f"Wrote {output} | frames={len(frames)} | fps={fps:.2f} | duration={duration:.1f}s")


if __name__ == "__main__":
    main()
