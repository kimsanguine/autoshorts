"""
Pipeline orchestrator for YouTube-to-Shorts conversion.
Manages the full workflow: download → transcribe → analyze → crop → subtitle → output.
Supports resume via progress.json.
"""

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Optional

from .config import (
    TEMP_DIR,
    OUTPUT_DIR,
    SEGMENT_SETTINGS,
    CROP_MODE,
    DEFAULT_SUBTITLE_STYLE,
    DEFAULT_FONT_PRESET,
    TRANSLATION_SETTINGS,
    get_timestamp,
)
from .downloader import VideoDownloader
from .transcriber import AudioTranscriber
from .segment_selector import SegmentSelector
from .video_cropper import VideoCropper
from .subtitle_overlay import SubtitleOverlay
from .subtitle_translator import SubtitleTranslator

logger = logging.getLogger(__name__)


class ShortsPipeline:
    """YouTube-to-Shorts 전체 파이프라인."""

    STEPS = [
        "download",
        "transcribe",
        "analyze",
        "crop",
        "subtitle",
        "finalize",
    ]

    def __init__(
        self,
        n_shorts: int = None,
        crop_mode: str = None,
        enable_subtitles: bool = True,
        subtitle_style: str = None,
        font_preset: str = None,
    ):
        self.n_shorts = n_shorts or SEGMENT_SETTINGS["n_shorts"]
        self.crop_mode = crop_mode or CROP_MODE
        self.enable_subtitles = enable_subtitles
        self.subtitle_style = subtitle_style or DEFAULT_SUBTITLE_STYLE
        self.font_preset = font_preset or DEFAULT_FONT_PRESET

    def generate(
        self,
        youtube_url: str,
        output_dir: Optional[Path] = None,
        keep_temp_files: bool = True,
        resume_from: Optional[Path] = None,
    ) -> dict:
        """
        YouTube URL에서 숏폼 영상들을 생성한다.

        Args:
            youtube_url: YouTube 영상 URL
            output_dir: 최종 출력 디렉토리 (None이면 output/)
            keep_temp_files: 중간 파일 보존 여부
            resume_from: 이전 run 디렉토리에서 재개

        Returns:
            dict with keys:
                output_paths: List of final MP4 paths
                metadata: Generation metadata dict
                run_dir: Temp run directory path
        """
        start_time = time.time()
        output_dir = Path(output_dir) if output_dir else OUTPUT_DIR

        # Resume or new run
        if resume_from:
            run_dir = Path(resume_from)
            progress = self._load_progress(run_dir)
            logger.info(f"Resuming from: {run_dir} (last step: {progress.get('last_step', 'none')})")
        else:
            run_dir = TEMP_DIR / f"run_{get_timestamp()}"
            progress = {"url": youtube_url, "steps": {}}
            run_dir.mkdir(parents=True, exist_ok=True)
            self._save_progress(run_dir, progress)

        try:
            # Step 1: Download
            if not self._step_done(progress, "download"):
                progress = self._step_download(youtube_url, run_dir, progress)
                self._save_progress(run_dir, progress)

            # Step 2: Transcribe
            if not self._step_done(progress, "transcribe"):
                progress = self._step_transcribe(run_dir, progress)
                self._save_progress(run_dir, progress)

            # Step 3: Analyze segments
            if not self._step_done(progress, "analyze"):
                progress = self._step_analyze(run_dir, progress)
                self._save_progress(run_dir, progress)

            # Step 4: Crop
            if not self._step_done(progress, "crop"):
                progress = self._step_crop(run_dir, progress)
                self._save_progress(run_dir, progress)

            # Step 5: Subtitle
            if not self._step_done(progress, "subtitle"):
                progress = self._step_subtitle(run_dir, progress)
                self._save_progress(run_dir, progress)

            # Step 6: Finalize
            result = self._step_finalize(run_dir, output_dir, progress)

            elapsed = time.time() - start_time
            logger.info(f"Pipeline complete in {elapsed:.1f}s. {len(result['output_paths'])} shorts generated.")

            # Cleanup
            if not keep_temp_files:
                shutil.rmtree(run_dir, ignore_errors=True)

            return result

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            progress["error"] = str(e)
            self._save_progress(run_dir, progress)
            raise

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _step_download(self, url: str, run_dir: Path, progress: dict) -> dict:
        logger.info("=" * 50)
        logger.info("Step 1/6: Download")
        logger.info("=" * 50)

        downloader = VideoDownloader()
        source_dir = run_dir / "source"
        result = downloader.download(url, output_dir=source_dir)

        progress["steps"]["download"] = {
            "status": "done",
            "video_path": str(result["video_path"]),
            "video_id": result["video_id"],
            "title": result["title"],
            "duration": result["duration"],
            "view_count": result.get("view_count", 0),
            "has_heatmap": result["heatmap"] is not None,
        }
        progress["last_step"] = "download"
        return progress

    def _step_transcribe(self, run_dir: Path, progress: dict) -> dict:
        logger.info("=" * 50)
        logger.info("Step 2/6: Transcribe")
        logger.info("=" * 50)

        video_path = Path(progress["steps"]["download"]["video_path"])
        transcript_dir = run_dir / "transcript"

        transcriber = AudioTranscriber()
        result = transcriber.transcribe(video_path, output_dir=transcript_dir)

        progress["steps"]["transcribe"] = {
            "status": "done",
            "words_path": str(result["words_path"]),
            "srt_path": str(result["srt_path"]),
            "word_count": len(result["words"]),
            "language": result["language"],
        }
        progress["last_step"] = "transcribe"
        return progress

    def _step_analyze(self, run_dir: Path, progress: dict) -> dict:
        logger.info("=" * 50)
        logger.info("Step 3/6: Analyze Segments")
        logger.info("=" * 50)

        # Load heatmap
        video_id = progress["steps"]["download"]["video_id"]
        info_path = run_dir / "source" / f"{video_id}.info.json"
        heatmap = None
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
            heatmap = info.get("heatmap")

        # Load transcript
        words_path = Path(progress["steps"]["transcribe"]["words_path"])
        with open(words_path) as f:
            transcript = json.load(f)

        duration = progress["steps"]["download"]["duration"]
        analysis_dir = run_dir / "analysis"

        selector = SegmentSelector(n_shorts=self.n_shorts)
        segments = selector.analyze(
            heatmap=heatmap,
            transcript=transcript,
            video_duration=duration,
            output_dir=analysis_dir,
        )

        progress["steps"]["analyze"] = {
            "status": "done",
            "segments_path": str(analysis_dir / "segments.json"),
            "n_segments": len(segments),
        }
        progress["last_step"] = "analyze"
        return progress

    def _step_crop(self, run_dir: Path, progress: dict) -> dict:
        logger.info("=" * 50)
        logger.info("Step 4/6: Crop to 9:16")
        logger.info("=" * 50)

        video_path = Path(progress["steps"]["download"]["video_path"])
        video_id = progress["steps"]["download"]["video_id"]

        segments_path = Path(progress["steps"]["analyze"]["segments_path"])
        with open(segments_path) as f:
            analysis = json.load(f)
        segments = analysis["segments"]

        cropped_dir = run_dir / "cropped"
        cropper = VideoCropper(crop_mode=self.crop_mode)
        results = cropper.crop_batch(
            video_path=video_path,
            segments=segments,
            output_dir=cropped_dir,
            video_id=video_id,
        )

        progress["steps"]["crop"] = {
            "status": "done",
            "cropped_paths": [str(r["cropped_path"]) for r in results],
            "n_cropped": len(results),
        }
        progress["last_step"] = "crop"
        return progress

    def _step_subtitle(self, run_dir: Path, progress: dict) -> dict:
        logger.info("=" * 50)
        logger.info("Step 5/6: Subtitle Overlay")
        logger.info("=" * 50)

        if not self.enable_subtitles:
            # 자막 없이 크롭된 영상 그대로 사용
            cropped_paths = progress["steps"]["crop"]["cropped_paths"]
            progress["steps"]["subtitle"] = {
                "status": "done",
                "output_paths": cropped_paths,
                "subtitles_enabled": False,
            }
            progress["last_step"] = "subtitle"
            return progress

        # Load segments and words
        segments_path = Path(progress["steps"]["analyze"]["segments_path"])
        with open(segments_path) as f:
            analysis = json.load(f)
        segments = analysis["segments"]

        words_path = Path(progress["steps"]["transcribe"]["words_path"])
        with open(words_path) as f:
            transcript = json.load(f)
        words = transcript["words"]

        cropped_paths = [Path(p) for p in progress["steps"]["crop"]["cropped_paths"]]
        srt_dir = run_dir / "subtitles"
        output_dir = run_dir / "output"

        # 언어 감지 → 번역 필요 여부 결정
        source_lang = progress["steps"]["transcribe"].get("language", "ko")
        translation_enabled = (
            TRANSLATION_SETTINGS["enabled"]
            and source_lang not in TRANSLATION_SETTINGS["skip_langs"]
        )

        translator = None
        if translation_enabled:
            translator = SubtitleTranslator()
            logger.info(f"번역 활성화: {source_lang} → {TRANSLATION_SETTINGS['target_lang']}")
        else:
            logger.info(f"번역 스킵: source_lang={source_lang}")

        # Build cropped_results for batch
        cropped_results = []
        for seg, cp in zip(segments, cropped_paths):
            cropped_results.append({"segment": seg, "cropped_path": cp})

        transcriber = AudioTranscriber()
        overlay = SubtitleOverlay(
            style=self.subtitle_style,
            font_preset=self.font_preset,
            translator=translator,
            source_lang=source_lang if translation_enabled else None,
        )
        results = overlay.apply_batch(
            cropped_results=cropped_results,
            transcriber=transcriber,
            words=words,
            output_dir=output_dir,
            srt_dir=srt_dir,
        )

        progress["steps"]["subtitle"] = {
            "status": "done",
            "output_paths": [str(r["output_path"]) for r in results],
            "subtitles_enabled": True,
            "translation_enabled": translation_enabled,
            "source_lang": source_lang,
        }
        progress["last_step"] = "subtitle"
        return progress

    def _step_finalize(self, run_dir: Path, output_dir: Path, progress: dict) -> dict:
        logger.info("=" * 50)
        logger.info("Step 6/6: Finalize")
        logger.info("=" * 50)

        output_dir.mkdir(parents=True, exist_ok=True)

        video_id = progress["steps"]["download"]["video_id"]
        source_paths = [Path(p) for p in progress["steps"]["subtitle"]["output_paths"]]

        # Load segments for metadata
        segments_path = Path(progress["steps"]["analyze"]["segments_path"])
        with open(segments_path) as f:
            analysis = json.load(f)
        segments = analysis["segments"]

        # Copy to output with final naming
        final_paths = []
        shorts_meta = []

        for i, (src, seg) in enumerate(zip(source_paths, segments)):
            final_name = f"{video_id}_short_{i+1:03d}.mp4"
            final_path = output_dir / final_name
            shutil.copy2(src, final_path)
            final_paths.append(final_path)

            shorts_meta.append({
                "index": i,
                "file": final_name,
                "source_start": seg["start"],
                "source_end": seg["end"],
                "score": seg["score"],
                "score_breakdown": seg.get("score_breakdown", {}),
                "reason": seg.get("reason", ""),
            })

            logger.info(f"  Output: {final_path}")

        # Generate metadata
        metadata = {
            "source": {
                "url": progress.get("url", ""),
                "title": progress["steps"]["download"]["title"],
                "video_id": video_id,
                "duration": progress["steps"]["download"]["duration"],
                "has_heatmap": progress["steps"]["download"].get("has_heatmap", False),
            },
            "settings": {
                "n_shorts": self.n_shorts,
                "crop_mode": self.crop_mode,
                "subtitles_enabled": self.enable_subtitles,
                "subtitle_style": self.subtitle_style,
                "font_preset": self.font_preset,
            },
            "shorts": shorts_meta,
        }

        metadata_path = output_dir / f"{video_id}_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"Metadata: {metadata_path}")

        progress["steps"]["finalize"] = {"status": "done"}
        progress["last_step"] = "finalize"
        self._save_progress(run_dir, progress)

        return {
            "output_paths": final_paths,
            "metadata": metadata,
            "metadata_path": metadata_path,
            "run_dir": run_dir,
        }

    # ------------------------------------------------------------------
    # Progress management
    # ------------------------------------------------------------------

    def _step_done(self, progress: dict, step: str) -> bool:
        return progress.get("steps", {}).get(step, {}).get("status") == "done"

    def _save_progress(self, run_dir: Path, progress: dict):
        path = run_dir / "progress.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False, indent=2, default=str)

    def _load_progress(self, run_dir: Path) -> dict:
        path = run_dir / "progress.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {"steps": {}}
