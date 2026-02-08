"""
YouTube video downloader with heatmap and metadata extraction.
Uses yt-dlp for downloading and extracting "Most Replayed" data.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from .config import DOWNLOAD_SETTINGS, TEMP_DIR, get_timestamp

logger = logging.getLogger(__name__)


class VideoDownloader:
    """YouTube 영상 다운로드 + 메타데이터/heatmap 추출."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir

    def download(
        self,
        url: str,
        output_dir: Optional[Path] = None,
    ) -> dict:
        """
        YouTube 영상을 다운로드하고 메타데이터를 추출한다.

        Args:
            url: YouTube URL
            output_dir: 다운로드 경로 (None이면 temp/run_TIMESTAMP/source/)

        Returns:
            dict with keys:
                video_path: Path to downloaded video
                info: Full metadata dict
                heatmap: List of heatmap entries or None
                video_id: YouTube video ID
                title: Video title
                duration: Video duration in seconds
        """
        import yt_dlp

        dest = output_dir or self.output_dir
        if dest is None:
            dest = TEMP_DIR / f"run_{get_timestamp()}" / "source"
        dest.mkdir(parents=True, exist_ok=True)

        video_id = self._extract_video_id(url)
        if not video_id:
            raise ValueError(f"Invalid YouTube URL: {url}")

        # Warn if already a shorts URL
        if "/shorts/" in url:
            logger.warning("이미 숏폼 URL입니다. 그래도 진행합니다.")

        ydl_opts = {
            "format": f"bestvideo[height<={DOWNLOAD_SETTINGS['max_resolution'].replace('p', '')}]+bestaudio/best",
            "merge_output_format": DOWNLOAD_SETTINGS["preferred_format"],
            "outtmpl": str(dest / "%(id)s.%(ext)s"),
            "writeinfojson": True,
            "quiet": True,
            "no_warnings": True,
        }

        # Auto subtitles
        if DOWNLOAD_SETTINGS["extract_auto_subs"]:
            ydl_opts["writeautomaticsub"] = True
            ydl_opts["subtitleslangs"] = DOWNLOAD_SETTINGS["sub_languages"]
            ydl_opts["subtitlesformat"] = "srt/vtt/best"

        logger.info(f"Downloading: {url}")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

        # Find downloaded video file
        video_path = self._find_video_file(dest, video_id)
        if not video_path:
            raise FileNotFoundError(f"Downloaded video not found in {dest}")

        # Extract heatmap
        heatmap = self._extract_heatmap(info)

        # Save info.json (yt-dlp may already do this, but ensure we have it)
        info_path = dest / f"{video_id}.info.json"
        if not info_path.exists():
            # yt-dlp의 info에는 직렬화 불가능한 객체가 있을 수 있으므로 안전하게 저장
            self._save_info_json(info, info_path)

        result = {
            "video_path": video_path,
            "info_path": info_path,
            "info": info,
            "heatmap": heatmap,
            "video_id": video_id,
            "title": info.get("title", "Unknown"),
            "duration": info.get("duration", 0),
            "view_count": info.get("view_count", 0),
            "chapters": info.get("chapters"),
        }

        logger.info(
            f"Download complete: {result['title']} "
            f"({result['duration']:.0f}s, "
            f"heatmap={'available' if heatmap else 'not available'})"
        )

        return result

    def _extract_video_id(self, url: str) -> Optional[str]:
        """YouTube URL에서 video ID 추출."""
        patterns = [
            r"(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})",
            r"(?:/shorts/)([a-zA-Z0-9_-]{11})",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def _find_video_file(self, directory: Path, video_id: str) -> Optional[Path]:
        """다운로드된 비디오 파일을 찾는다."""
        video_extensions = [".mp4", ".mkv", ".webm"]
        for ext in video_extensions:
            candidate = directory / f"{video_id}{ext}"
            if candidate.exists():
                return candidate
        # Fallback: 디렉토리에서 비디오 파일 검색
        for ext in video_extensions:
            files = list(directory.glob(f"*{ext}"))
            # info.json 제외
            video_files = [f for f in files if ".info." not in f.name]
            if video_files:
                return video_files[0]
        return None

    def _extract_heatmap(self, info: dict) -> Optional[list]:
        """
        yt-dlp info에서 heatmap 데이터를 추출한다.

        Returns:
            List of {"start_time": float, "end_time": float, "value": float}
            or None if not available.
        """
        heatmap = info.get("heatmap")
        if heatmap:
            logger.info(f"Heatmap found: {len(heatmap)} data points")
            return heatmap

        # Fallback: chapters에서 힌트 추출
        chapters = info.get("chapters")
        if chapters:
            logger.info(f"No heatmap, but {len(chapters)} chapters found")

        return None

    def _save_info_json(self, info: dict, path: Path):
        """info dict를 JSON으로 안전하게 저장."""
        def _serializable(obj):
            if isinstance(obj, Path):
                return str(obj)
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)

        safe_info = {}
        # 직렬화 가능한 키만 추출
        keys_to_save = [
            "id", "title", "description", "duration", "view_count",
            "like_count", "comment_count", "upload_date", "channel",
            "channel_id", "categories", "tags", "chapters", "heatmap",
            "subtitles", "automatic_captions", "thumbnail", "webpage_url",
        ]
        for key in keys_to_save:
            if key in info:
                safe_info[key] = info[key]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(safe_info, f, ensure_ascii=False, indent=2, default=str)
