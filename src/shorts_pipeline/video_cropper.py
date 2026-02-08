"""
Video cropping module for 16:9 → 9:16 vertical conversion.
Extracts segments and crops to vertical format using FFmpeg.
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional

from .config import VIDEO_SETTINGS, CROP_MODE

logger = logging.getLogger(__name__)


class VideoCropper:
    """16:9 영상을 9:16 세로로 크롭하고 구간을 추출한다."""

    def __init__(self, crop_mode: Optional[str] = None):
        self.crop_mode = crop_mode or CROP_MODE
        self.target_w, self.target_h = VIDEO_SETTINGS["resolution"]  # 1080, 1920
        self.fps = VIDEO_SETTINGS["fps"]
        self.video_bitrate = VIDEO_SETTINGS["video_bitrate"]
        self.audio_bitrate = VIDEO_SETTINGS["audio_bitrate"]

    def crop_segment(
        self,
        video_path: Path,
        start: float,
        end: float,
        output_path: Path,
    ) -> Path:
        """
        원본 영상에서 구간을 추출하고 9:16으로 크롭한다.

        Args:
            video_path: 원본 영상
            start: 시작 시간 (초)
            end: 끝 시간 (초)
            output_path: 출력 경로

        Returns:
            출력 파일 경로
        """
        video_path = Path(video_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        duration = end - start

        logger.info(
            f"Cropping {video_path.name}: {start:.1f}s - {end:.1f}s "
            f"({duration:.1f}s) → {output_path.name}"
        )

        # Center crop: ih*9/16 너비로 중앙 크롭 후 1080x1920으로 스케일
        # crop=out_w:out_h:x:y → crop=ih*9/16:ih:(iw-ih*9/16)/2:0
        crop_filter = "crop=ih*9/16:ih:(iw-ih*9/16)/2:0"
        scale_filter = f"scale={self.target_w}:{self.target_h}"

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", str(video_path),
            "-t", str(duration),
            "-vf", f"{crop_filter},{scale_filter}",
            "-c:v", "libx264",
            "-preset", "medium",
            "-b:v", self.video_bitrate,
            "-c:a", "aac",
            "-b:a", self.audio_bitrate,
            "-r", str(self.fps),
            "-movflags", "+faststart",
            str(output_path),
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
        )

        if result.returncode != 0:
            logger.error(f"FFmpeg crop failed: {result.stderr[-500:]}")
            raise RuntimeError(f"FFmpeg crop failed for {output_path.name}")

        logger.info(f"Cropped: {output_path.name}")
        return output_path

    def crop_batch(
        self,
        video_path: Path,
        segments: list,
        output_dir: Path,
        video_id: str = "short",
    ) -> list:
        """
        여러 구간을 일괄 크롭한다.

        Args:
            video_path: 원본 영상
            segments: [{"start": float, "end": float, ...}, ...]
            output_dir: 출력 디렉토리
            video_id: 파일명 접두사

        Returns:
            List of {"segment": dict, "cropped_path": Path}
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for i, seg in enumerate(segments):
            filename = f"{video_id}_short_{i+1:03d}.mp4"
            output_path = output_dir / filename

            cropped = self.crop_segment(
                video_path=video_path,
                start=seg["start"],
                end=seg["end"],
                output_path=output_path,
            )

            results.append({
                "segment": seg,
                "cropped_path": cropped,
            })

        logger.info(f"Batch crop complete: {len(results)} clips")
        return results
