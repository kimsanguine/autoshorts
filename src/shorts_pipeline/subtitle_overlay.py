"""
Subtitle overlay module.
Burns subtitles into cropped video clips using FFmpeg.
Supports 3 styles (box, outline, highlight) with subtitles→drawtext→copy fallback.
"""

import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

from .config import (
    VIDEO_SETTINGS,
    FONT_PRESETS,
    SUBTITLE_STYLES,
    DEFAULT_FONT_PRESET,
    DEFAULT_SUBTITLE_STYLE,
    FONT_DIR,
    TRANSLATION_SETTINGS,
    BILINGUAL_SETTINGS,
)

logger = logging.getLogger(__name__)


class SubtitleOverlay:
    """FFmpeg 자막 오버레이. 3종 스타일 + subtitles→drawtext→copy 이중 접근."""

    def __init__(
        self,
        style: Optional[str] = None,
        font_preset: Optional[str] = None,
        translator=None,
        source_lang: Optional[str] = None,
    ):
        self.style_name = style or DEFAULT_SUBTITLE_STYLE
        self.style = SUBTITLE_STYLES.get(self.style_name, SUBTITLE_STYLES[DEFAULT_SUBTITLE_STYLE])

        # 폰트 설정
        preset_key = font_preset or DEFAULT_FONT_PRESET
        self.font_preset = FONT_PRESETS.get(preset_key, FONT_PRESETS[DEFAULT_FONT_PRESET])
        self.font_path = Path(self.font_preset["path"])
        self.font_name = self.font_preset["name"]

        # 비디오 설정
        self.fontsize = VIDEO_SETTINGS["subtitle_fontsize"]
        self.color = VIDEO_SETTINGS["subtitle_color"]
        self.margin = VIDEO_SETTINGS["subtitle_margin"]
        self.video_bitrate = VIDEO_SETTINGS["video_bitrate"]
        self.audio_bitrate = VIDEO_SETTINGS["audio_bitrate"]

        # 번역 설정
        self.translator = translator
        self.source_lang = source_lang
        self._bilingual = (
            translator is not None
            and source_lang is not None
            and source_lang not in TRANSLATION_SETTINGS["skip_langs"]
        )

        self._has_subtitles_filter = self._check_subtitles_filter()
        if self._has_subtitles_filter:
            logger.info("FFmpeg subtitles filter 사용 가능 (libass)")
        else:
            logger.info("drawtext fallback 사용 (libass 미설치)")
        if self._bilingual:
            logger.info(f"이중 자막 모드 활성화 ({source_lang} + ko)")

    def apply(
        self,
        video_path: Path,
        srt_path: Path,
        output_path: Path,
        words: Optional[list] = None,
    ) -> Path:
        """
        비디오에 자막을 오버레이한다.

        Args:
            video_path: 입력 비디오 (크롭 완료)
            srt_path: SRT 자막 파일
            output_path: 출력 경로
            words: word-level timestamps (highlight 스타일용)

        Returns:
            출력 파일 경로
        """
        video_path = Path(video_path)
        srt_path = Path(srt_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not srt_path.exists() or srt_path.stat().st_size == 0:
            logger.warning(f"SRT 파일 없음: {srt_path}. 원본 복사.")
            return self._copy_video(video_path, output_path)

        logger.info(f"자막 오버레이: {video_path.name} + {srt_path.name} (style={self.style_name})")

        # 이중 자막 모드 (번역 필요 시)
        if self._bilingual:
            try:
                return self._apply_drawtext_bilingual(video_path, srt_path, output_path)
            except Exception as e:
                logger.warning(f"이중 자막 실패, 단일 자막으로 fallback: {e}")

        # highlight 스타일이고 words가 있으면 word-level highlight 적용
        if self.style_name == "highlight" and words:
            try:
                return self._apply_highlight_style(video_path, words, output_path)
            except Exception as e:
                logger.warning(f"Highlight 스타일 실패, SRT 기반으로 fallback: {e}")

        # 1차: drawtext filter (fontfile 직접 지정 — 폰트 이름 불일치 문제 없음)
        try:
            return self._apply_drawtext_fallback(video_path, srt_path, output_path)
        except Exception as e:
            logger.warning(f"drawtext 필터 실패, subtitles로 fallback: {e}")

        # 2차: subtitles filter (ASS, libass — fontsdir/FontName 매칭 필요)
        if self._has_subtitles_filter:
            try:
                return self._apply_subtitles_method(video_path, srt_path, output_path)
            except Exception as e:
                logger.warning(f"subtitles 필터 실패, 원본 복사: {e}")

        # 3차: 원본 복사
        return self._copy_video(video_path, output_path)

    def apply_batch(
        self,
        cropped_results: list,
        transcriber,
        words: list,
        output_dir: Path,
        srt_dir: Path,
    ) -> list:
        """
        여러 크롭된 클립에 자막을 일괄 오버레이한다.

        Args:
            cropped_results: [{"segment": dict, "cropped_path": Path}, ...]
            transcriber: AudioTranscriber 인스턴스 (generate_segment_srt용)
            words: 전체 word-level timestamps
            output_dir: 최종 출력 디렉토리
            srt_dir: SRT 파일 저장 디렉토리

        Returns:
            List of {"segment": dict, "output_path": Path}
        """
        output_dir = Path(output_dir)
        srt_dir = Path(srt_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        srt_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for item in cropped_results:
            seg = item["segment"]
            cropped_path = item["cropped_path"]

            # 구간별 SRT 생성 (시간 offset 적용)
            srt_path = srt_dir / f"{cropped_path.stem}.srt"
            transcriber.generate_segment_srt(
                words=words,
                start_time=seg["start"],
                end_time=seg["end"],
                output_path=srt_path,
            )

            # 구간별 words 추출 (highlight 스타일용)
            segment_words = None
            if self.style_name == "highlight":
                segment_words = [
                    {
                        **w,
                        "start": w.get("start", 0) - seg["start"],
                        "end": w.get("end", 0) - seg["start"],
                    }
                    for w in words
                    if w.get("start", 0) >= seg["start"] and w.get("end", 0) <= seg["end"]
                ]

            # 자막 오버레이
            output_path = output_dir / cropped_path.name
            self.apply(
                video_path=cropped_path,
                srt_path=srt_path,
                output_path=output_path,
                words=segment_words,
            )

            results.append({
                "segment": seg,
                "output_path": output_path,
            })

        logger.info(f"일괄 자막 오버레이 완료: {len(results)}개 클립")
        return results

    # ------------------------------------------------------------------
    # 렌더링 방식
    # ------------------------------------------------------------------

    def _apply_subtitles_method(
        self, video_path: Path, srt_path: Path, output_path: Path,
    ) -> Path:
        """ASS subtitles 필터로 자막 렌더링 (1차)."""
        srt_escaped = self._escape_ffmpeg_path(str(srt_path.absolute()))

        # force_style 구성
        ass_params = self.style.get("ass_style", {})
        force_style_parts = [
            f"FontName={self.font_name}",
            f"FontSize={self.fontsize}",
            f"PrimaryColour=&H00FFFFFF",
            f"MarginV={self.margin}",
            "Alignment=2",
        ]
        for key, val in ass_params.items():
            force_style_parts.append(f"{key}={val}")
        force_style = ",".join(force_style_parts)

        # fontsdir: 폰트 파일이 존재할 때만 포함
        filter_parts = [f"subtitles='{srt_escaped}'"]
        if self.font_path.exists():
            fontsdir = self._escape_ffmpeg_path(str(self.font_path.parent.absolute()))
            filter_parts.append(f"fontsdir='{fontsdir}'")
        filter_parts.append(f"force_style='{force_style}'")
        subtitle_filter = ":".join(filter_parts)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", subtitle_filter,
            "-c:v", "libx264",
            "-preset", "medium",
            "-b:v", self.video_bitrate,
            "-c:a", "copy",
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"subtitles 필터 실패: {result.stderr[-500:]}")

        logger.info(f"subtitles 필터 완료: {output_path.name}")
        return output_path

    def _apply_drawtext_fallback(
        self, video_path: Path, srt_path: Path, output_path: Path,
    ) -> Path:
        """drawtext 필터로 자막 렌더링 (2차 fallback)."""
        entries = self._parse_srt_entries(srt_path)
        if not entries:
            logger.warning(f"SRT 항목 없음: {srt_path}")
            return self._copy_video(video_path, output_path)

        # 폰트 파일 확인
        fontfile = str(self.font_path.absolute()) if self.font_path.exists() else None
        if not fontfile:
            logger.warning(f"폰트 파일 없음: {self.font_path}. 시스템 기본 사용.")

        # 스타일 파라미터
        dt_style = self.style.get("drawtext_style", {})

        # drawtext 필터 체인 구성
        filters = []
        for entry in entries:
            safe_text = self._escape_drawtext_text(entry["text"])

            parts = [
                f"text='{safe_text}'",
                f"fontsize={self.fontsize}",
                f"fontcolor={self.color}",
                f"x=(w-text_w)/2",
                f"y=h-{self.margin}-text_h",
                f"enable='between(t,{entry['start']:.3f},{entry['end']:.3f})'",
            ]

            if fontfile:
                escaped_fontfile = self._escape_ffmpeg_path(fontfile)
                parts.append(f"fontfile='{escaped_fontfile}'")

            # 스타일별 파라미터 추가
            for key, val in dt_style.items():
                parts.append(f"{key}={val}")

            filters.append("drawtext=" + ":".join(parts))

        vf = ",".join(filters)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", "medium",
            "-b:v", self.video_bitrate,
            "-c:a", "copy",
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"drawtext 필터 실패: {result.stderr[-500:]}")

        logger.info(f"drawtext fallback 완료: {output_path.name}")
        return output_path

    def _apply_drawtext_bilingual(
        self, video_path: Path, srt_path: Path, output_path: Path,
    ) -> Path:
        """이중 자막 렌더링: 원문(위, 작은 글씨) + 한국어 번역(아래, 큰 글씨)."""
        entries = self._parse_srt_entries(srt_path)
        if not entries:
            logger.warning(f"SRT 항목 없음: {srt_path}")
            return self._copy_video(video_path, output_path)

        # 번역 실행
        translated = self.translator.translate_entries(entries, self.source_lang)

        # 폰트 파일 확인
        fontfile = str(self.font_path.absolute()) if self.font_path.exists() else None
        if not fontfile:
            logger.warning(f"폰트 파일 없음: {self.font_path}. 시스템 기본 사용.")

        # 스타일 파라미터
        dt_style = self.style.get("drawtext_style", {})

        # 이중 자막 크기 계산
        bi = BILINGUAL_SETTINGS
        fontsize_orig = int(self.fontsize * bi["original_fontsize_ratio"])
        fontsize_trans = int(self.fontsize * bi["translated_fontsize_ratio"])
        line_gap = bi["line_gap"]

        # 위치 계산:
        # 한국어(아래): y = h - margin - text_h  (기존과 동일 위치)
        # 영어(위):     y = h - margin - text_h_ko - gap - text_h_en
        # drawtext에서 text_h는 현재 텍스트의 높이이므로 각각 다른 fontsize의 text_h 사용
        # 한국어 y: h - margin - text_h
        # 영어 y:   h - margin - fontsize_trans - line_gap - text_h
        y_translated = f"h-{self.margin}-text_h"
        y_original = f"h-{self.margin}-{fontsize_trans}-{line_gap}-text_h"

        filters = []
        for entry in translated:
            has_translation = bool(entry.get("translated"))

            if has_translation:
                # 한국어 번역 (아래, 큰 글씨)
                safe_trans = self._escape_drawtext_text(entry["translated"])
                trans_parts = [
                    f"text='{safe_trans}'",
                    f"fontsize={fontsize_trans}",
                    f"fontcolor={self.color}",
                    f"x=(w-text_w)/2",
                    f"y={y_translated}",
                    f"enable='between(t,{entry['start']:.3f},{entry['end']:.3f})'",
                ]
                if fontfile:
                    trans_parts.append(f"fontfile='{self._escape_ffmpeg_path(fontfile)}'")
                for key, val in dt_style.items():
                    trans_parts.append(f"{key}={val}")
                filters.append("drawtext=" + ":".join(trans_parts))

                # 영어 원문 (위, 작은 글씨)
                safe_orig = self._escape_drawtext_text(entry["original"])
                orig_parts = [
                    f"text='{safe_orig}'",
                    f"fontsize={fontsize_orig}",
                    f"fontcolor={self.color}@0.85",
                    f"x=(w-text_w)/2",
                    f"y={y_original}",
                    f"enable='between(t,{entry['start']:.3f},{entry['end']:.3f})'",
                ]
                if fontfile:
                    orig_parts.append(f"fontfile='{self._escape_ffmpeg_path(fontfile)}'")
                for key, val in dt_style.items():
                    orig_parts.append(f"{key}={val}")
                filters.append("drawtext=" + ":".join(orig_parts))
            else:
                # 번역 없으면 원문만 기본 위치에 표시
                safe_text = self._escape_drawtext_text(entry["original"])
                parts = [
                    f"text='{safe_text}'",
                    f"fontsize={self.fontsize}",
                    f"fontcolor={self.color}",
                    f"x=(w-text_w)/2",
                    f"y=h-{self.margin}-text_h",
                    f"enable='between(t,{entry['start']:.3f},{entry['end']:.3f})'",
                ]
                if fontfile:
                    parts.append(f"fontfile='{self._escape_ffmpeg_path(fontfile)}'")
                for key, val in dt_style.items():
                    parts.append(f"{key}={val}")
                filters.append("drawtext=" + ":".join(parts))

        if not filters:
            return self._copy_video(video_path, output_path)

        vf = ",".join(filters)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", "medium",
            "-b:v", self.video_bitrate,
            "-c:a", "copy",
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"이중 자막 drawtext 실패: {result.stderr[-500:]}")

        logger.info(f"이중 자막 완료: {output_path.name}")
        return output_path

    def _apply_highlight_style(
        self, video_path: Path, words: list, output_path: Path,
    ) -> Path:
        """
        word-level highlight 스타일.
        각 단어를 개별 drawtext로 표시. 현재 발화 중인 단어만 강조색.
        청크 단위로 표시하되, 개별 단어의 색상이 시간에 따라 변경.
        """
        if not words:
            raise ValueError("highlight 스타일에 words가 필요합니다")

        highlight_color = self.style.get("highlight_color", "yellow")
        dt_style = self.style.get("drawtext_style", {})

        # 폰트
        fontfile = str(self.font_path.absolute()) if self.font_path.exists() else None

        # words를 청크로 묶기
        chunks = self._group_words_for_highlight(words)
        filters = []

        for chunk in chunks:
            if not chunk["words"]:
                continue

            # 각 단어의 발화 시간에 따라 색상 전환
            # 접근: 청크 전체 표시(흰색) + 현재 단어만 강조색 오버레이
            # 간소화: 각 단어 타이밍마다 전체 텍스트를 다시 그리되 현재 단어만 강조
            for wi, active_word in enumerate(chunk["words"]):
                # 이 단어가 발화되는 시간 동안: 전체 청크 텍스트 표시
                # 강조 단어 앞/뒤 분리하여 색상 다르게 표현
                w_start = active_word.get("start", chunk["start"])
                w_end = active_word.get("end", chunk["end"])

                # 전체 청크 텍스트 (흰색 base)
                chunk_text = " ".join(w["word"] for w in chunk["words"])
                safe_text = self._escape_drawtext_text(chunk_text)

                base_parts = [
                    f"text='{safe_text}'",
                    f"fontsize={self.fontsize}",
                    f"fontcolor={self.color}",
                    f"x=(w-text_w)/2",
                    f"y=h-{self.margin}-text_h",
                    f"enable='between(t\\,{w_start:.3f}\\,{w_end:.3f})'",
                ]
                if fontfile:
                    base_parts.append(f"fontfile='{self._escape_ffmpeg_path(fontfile)}'")
                for key, val in dt_style.items():
                    base_parts.append(f"{key}={val}")
                filters.append("drawtext=" + ":".join(base_parts))

                # 강조 단어만 덮어씌우기 (같은 위치에 강조색)
                # x 오프셋: 앞 단어들의 텍스트 길이만큼 이동
                # drawtext의 text_w는 현재 text의 너비이므로 이를 활용할 수 없음
                # → 대신 전체 청크를 강조색으로 다시 그리되 alpha trick 사용 불가
                # → 간소화: 강조 단어만 별도 표시 (전체 텍스트 대신 강조 단어만)
                active_text = self._escape_drawtext_text(active_word["word"])
                h_parts = [
                    f"text='{active_text}'",
                    f"fontsize={self.fontsize}",
                    f"fontcolor={highlight_color}",
                    f"x=(w-text_w)/2",  # 단어 하나만이므로 중앙 정렬
                    f"y=h-{self.margin}-text_h-{self.fontsize + 5}",  # 메인 자막 위에 표시
                    f"enable='between(t\\,{w_start:.3f}\\,{w_end:.3f})'",
                ]
                if fontfile:
                    h_parts.append(f"fontfile='{self._escape_ffmpeg_path(fontfile)}'")
                h_parts.append("borderw=2")
                h_parts.append("bordercolor=black")
                filters.append("drawtext=" + ":".join(h_parts))

        if not filters:
            raise ValueError("highlight 필터 생성 실패")

        vf = ",".join(filters)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", "medium",
            "-b:v", self.video_bitrate,
            "-c:a", "copy",
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"highlight 스타일 실패: {result.stderr[-500:]}")

        logger.info(f"highlight 스타일 완료: {output_path.name}")
        return output_path

    # ------------------------------------------------------------------
    # 유틸리티
    # ------------------------------------------------------------------

    def _check_subtitles_filter(self) -> bool:
        """FFmpeg subtitles 필터 사용 가능 여부 확인."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-filters"],
                capture_output=True, text=True, timeout=10,
            )
            return "subtitles" in result.stdout
        except Exception:
            return False

    def _parse_srt_entries(self, srt_path: Path) -> list:
        """SRT 파일 → [{start, end, text}, ...] 파싱."""
        try:
            content = srt_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"SRT 읽기 실패: {e}")
            return []

        entries = []
        blocks = re.split(r"\n\s*\n", content.strip())

        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) < 3:
                continue

            ts_match = re.match(
                r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})",
                lines[1].strip(),
            )
            if not ts_match:
                continue

            start = self._srt_time_to_seconds(ts_match.group(1))
            end = self._srt_time_to_seconds(ts_match.group(2))
            text = " ".join(line.strip() for line in lines[2:] if line.strip())

            if text:
                entries.append({"start": start, "end": end, "text": text})

        return entries

    def _srt_time_to_seconds(self, time_str: str) -> float:
        """SRT 타임스탬프 (HH:MM:SS,mmm) → 초 변환."""
        time_str = time_str.replace(",", ".")
        parts = time_str.split(":")
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])

    def _escape_ffmpeg_path(self, path: str) -> str:
        """FFmpeg 필터 구문용 경로 이스케이프."""
        return path.replace("\\", "/").replace(":", "\\:")

    def _escape_drawtext_text(self, text: str) -> str:
        """drawtext 필터용 텍스트 이스케이프."""
        return (
            text
            .replace("\\", "\\\\")
            .replace("'", "'\\''")
            .replace(":", "\\:")
            .replace("%", "%%")
        )

    def _copy_video(self, src: Path, dst: Path) -> Path:
        """원본 비디오를 그대로 복사 (3차 fallback)."""
        subprocess.run(["cp", str(src), str(dst)], check=True)
        logger.info(f"원본 복사: {dst.name}")
        return dst

    def _group_words_for_highlight(self, words: list) -> list:
        """highlight 스타일용 단어 그룹핑 (3-5단어씩)."""
        chunks = []
        current = []
        max_words = 4

        for word in words:
            current.append(word)
            if len(current) >= max_words:
                chunks.append({
                    "words": current,
                    "start": current[0].get("start", 0),
                    "end": current[-1].get("end", 0),
                })
                current = []

        if current:
            if len(current) < 2 and chunks:
                # 잔여분 병합
                chunks[-1]["words"].extend(current)
                chunks[-1]["end"] = current[-1].get("end", chunks[-1]["end"])
            else:
                chunks.append({
                    "words": current,
                    "start": current[0].get("start", 0),
                    "end": current[-1].get("end", 0),
                })

        return chunks
