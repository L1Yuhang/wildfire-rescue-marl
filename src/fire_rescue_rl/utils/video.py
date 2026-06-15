"""GIF/MP4 export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np


def save_frames(frames: Iterable[np.ndarray], path: str | Path, fps: int = 6) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = list(frames)
    if not frames:
        raise ValueError("No frames to save")
    if output.suffix.lower() == ".gif":
        imageio.mimsave(output, frames, duration=1.0 / fps)
    elif output.suffix.lower() == ".mp4":
        try:
            imageio.mimsave(output, frames, fps=fps, macro_block_size=1)
        except ValueError:
            _save_mp4_cv2(frames, output, fps)
    else:
        raise ValueError(f"Unsupported format {output.suffix}")
    return output


def _save_mp4_cv2(frames: list[np.ndarray], output: Path, fps: int) -> None:
    import cv2

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open MP4 writer for {output}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

