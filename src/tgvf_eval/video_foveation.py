from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from tgvf_eval.config import validate_video_foveation_mode


@dataclass
class VideoFoveationStats:
    video_foveation_mode: str
    video_nframes: int
    nextframe_reuse_ttl: int | None
    num_reencode_events: int = 0
    num_reuse_encode_events: int = 0
    avg_foveation_events_per_query: float = 0.0

    @property
    def reuse_rate(self) -> float:
        denom = self.num_reencode_events + self.num_reuse_encode_events
        return self.num_reuse_encode_events / max(1, denom)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_foveation_mode": self.video_foveation_mode,
            "video_nframes": self.video_nframes,
            "nextframe_reuse_ttl": self.nextframe_reuse_ttl,
            "num_reencode_events": self.num_reencode_events,
            "num_reuse_encode_events": self.num_reuse_encode_events,
            "reuse_rate": self.reuse_rate,
            "avg_foveation_events_per_query": self.avg_foveation_events_per_query,
        }


class VideoFoveationController:
    """Small v0 state machine for mocked/real video FVT update accounting."""

    def __init__(
        self,
        *,
        mode: str,
        nextframe_reuse_ttl: int = 1,
        target_generator: Callable[[int, Any], Any],
        fvt_encoder: Callable[[Any, Any], Any],
    ) -> None:
        validate_video_foveation_mode(mode)
        if nextframe_reuse_ttl < 1:
            raise ValueError("nextframe_reuse_ttl must be >= 1")
        self.mode = mode
        self.nextframe_reuse_ttl = nextframe_reuse_ttl
        self.target_generator = target_generator
        self.fvt_encoder = fvt_encoder

    def process_frames(self, frames: list[Any]) -> tuple[list[Any], VideoFoveationStats]:
        stats = VideoFoveationStats(
            video_foveation_mode=self.mode,
            video_nframes=len(frames),
            nextframe_reuse_ttl=self.nextframe_reuse_ttl if self.mode == "nextframe_encode_no_reencode" else None,
        )
        if self.mode == "off":
            return [], stats

        outputs = []
        cached_hq = None
        reuse_remaining = 0
        for index, frame in enumerate(frames):
            if self.mode == "per_frame_reencode" or cached_hq is None or reuse_remaining <= 0:
                cached_hq = self.target_generator(index, frame)
                stats.num_reencode_events += 1
                reuse_remaining = self.nextframe_reuse_ttl
            else:
                stats.num_reuse_encode_events += 1
                reuse_remaining -= 1
            outputs.append(self.fvt_encoder(cached_hq, frame))
        stats.avg_foveation_events_per_query = float(stats.num_reencode_events + stats.num_reuse_encode_events)
        return outputs, stats
