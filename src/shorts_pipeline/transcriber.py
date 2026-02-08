"""
Audio transcription with word-level timestamps using WhisperX.
Generates word-level JSON and SRT subtitle files.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import torch
# PyTorch 2.6+에서 pyannote/lightning_fabric VAD 모델 로딩 시 weights_only=True 문제 우회
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from .config import TRANSCRIPTION_SETTINGS, TEMP_DIR, get_timestamp

logger = logging.getLogger(__name__)


class AudioTranscriber:
    """WhisperX 기반 word-level 트랜스크립션."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        language: Optional[str] = None,
    ):
        self.model_name = model_name or TRANSCRIPTION_SETTINGS["model"]
        self.device = device or TRANSCRIPTION_SETTINGS["device"]
        self.compute_type = compute_type or TRANSCRIPTION_SETTINGS["compute_type"]
        self.language = language or TRANSCRIPTION_SETTINGS["language"]
        self.batch_size = TRANSCRIPTION_SETTINGS["batch_size"]
        self._model = None
        self._align_model = None
        self._align_metadata = None

    def transcribe(
        self,
        audio_path: Path,
        output_dir: Optional[Path] = None,
    ) -> dict:
        """
        오디오/비디오 파일을 word-level로 트랜스크립션한다.

        Args:
            audio_path: 오디오 또는 비디오 파일 경로
            output_dir: 결과 저장 경로 (None이면 자동 생성)

        Returns:
            dict with keys:
                segments: List of segment dicts with word-level timestamps
                words: Flat list of all words with timestamps
                text: Full transcript text
                srt_path: Path to generated SRT file
                words_path: Path to words JSON file
                language: Detected language
        """
        import whisperx

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        dest = output_dir
        if dest is None:
            dest = audio_path.parent.parent / "transcript"
        dest.mkdir(parents=True, exist_ok=True)

        # Step 1: Load audio
        logger.info(f"Loading audio: {audio_path.name}")
        audio = whisperx.load_audio(str(audio_path))

        # Step 2: Transcribe with Whisper
        logger.info(f"Transcribing with {self.model_name} on {self.device}...")
        if self._model is None:
            self._model = whisperx.load_model(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                language=self.language,
            )

        result = self._model.transcribe(
            audio,
            batch_size=self.batch_size,
            language=self.language,
        )

        detected_language = result.get("language", self.language)
        logger.info(f"Transcription complete. Language: {detected_language}, Segments: {len(result['segments'])}")

        # Step 3: Align for word-level timestamps
        logger.info("Aligning for word-level timestamps...")
        if self._align_model is None:
            self._align_model, self._align_metadata = whisperx.load_align_model(
                language_code=detected_language,
                device=self.device,
            )

        aligned = whisperx.align(
            result["segments"],
            self._align_model,
            self._align_metadata,
            audio,
            device=self.device,
            return_char_alignments=False,
        )

        segments = aligned.get("segments", result["segments"])

        # Extract flat word list
        words = self._extract_words(segments)
        full_text = " ".join(w["word"] for w in words)

        logger.info(f"Alignment complete. Words: {len(words)}")

        # Step 4: Save outputs
        stem = audio_path.stem

        # Save words JSON
        words_path = dest / f"{stem}_words.json"
        with open(words_path, "w", encoding="utf-8") as f:
            json.dump({
                "language": detected_language,
                "word_count": len(words),
                "words": words,
                "segments": [self._clean_segment(s) for s in segments],
            }, f, ensure_ascii=False, indent=2)

        # Save full SRT
        srt_path = dest / f"{stem}.srt"
        self._write_srt(segments, srt_path)

        logger.info(f"Saved: {words_path.name}, {srt_path.name}")

        return {
            "segments": segments,
            "words": words,
            "text": full_text,
            "srt_path": srt_path,
            "words_path": words_path,
            "language": detected_language,
        }

    def generate_segment_srt(
        self,
        words: list,
        start_time: float,
        end_time: float,
        output_path: Path,
    ) -> Path:
        """
        전체 words 리스트에서 특정 구간의 SRT를 생성한다.
        타임스탬프를 0 기준으로 offset 조정.

        Args:
            words: 전체 word-level timestamps
            start_time: 구간 시작 (초)
            end_time: 구간 끝 (초)
            output_path: SRT 저장 경로

        Returns:
            Path to generated SRT file
        """
        # Filter words in range
        segment_words = [
            w for w in words
            if w.get("start", 0) >= start_time and w.get("end", 0) <= end_time
        ]

        if not segment_words:
            # Fallback: 근처 단어라도 포함
            segment_words = [
                w for w in words
                if w.get("start", 0) >= start_time - 0.5 and w.get("end", 0) <= end_time + 0.5
            ]

        # Group words into subtitle chunks (자연어 분할, max 5 words)
        chunks = self._group_words_into_chunks(segment_words, max_words=5, min_words=2)

        # 각 chunk의 시작/끝 시간 계산 (offset 적용)
        timings = []
        for chunk in chunks:
            if not chunk:
                continue
            cs = max(0, chunk[0].get("start", start_time) - start_time)
            ce = max(cs + 0.1, chunk[-1].get("end", end_time) - start_time)
            timings.append({"chunk": chunk, "start": cs, "end": ce})

        # 자막 끊김 방지: 각 항목의 end를 다음 항목의 start까지 연장
        for i in range(len(timings) - 1):
            next_start = timings[i + 1]["start"]
            if next_start > timings[i]["end"]:
                timings[i]["end"] = next_start

        # Write SRT
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for i, t in enumerate(timings):
                text = " ".join(w["word"] for w in t["chunk"])
                f.write(f"{i + 1}\n")
                f.write(f"{self._format_timestamp(t['start'])} --> {self._format_timestamp(t['end'])}\n")
                f.write(f"{text}\n\n")

        return output_path

    def _extract_words(self, segments: list) -> list:
        """segments에서 word-level 데이터를 flat list로 추출."""
        words = []
        for seg in segments:
            seg_words = seg.get("words", [])
            for w in seg_words:
                if "word" in w:
                    words.append({
                        "word": w["word"].strip(),
                        "start": w.get("start", seg.get("start", 0)),
                        "end": w.get("end", seg.get("end", 0)),
                        "score": w.get("score", 0),
                    })
        return words

    def _clean_segment(self, segment: dict) -> dict:
        """segment dict를 JSON-serializable하게 정리."""
        return {
            "start": segment.get("start", 0),
            "end": segment.get("end", 0),
            "text": segment.get("text", ""),
            "words": [
                {
                    "word": w.get("word", "").strip(),
                    "start": w.get("start", 0),
                    "end": w.get("end", 0),
                    "score": w.get("score", 0),
                }
                for w in segment.get("words", [])
                if "word" in w
            ],
        }

    def _group_words_into_chunks(
        self, words: list, max_words: int = 5, min_words: int = 2,
        pause_threshold: float = 0.3,
    ) -> list:
        """
        단어들을 자연스러운 자막 청크로 그룹핑.

        분할 우선순위:
        1. 문장 종결 (. ! ?) → min_words 이상이면 즉시 분할
        2. 절 구분 (, ; :) + 발화 pause → 분할
        3. 단독 pause (pause_threshold 초 이상) → 분할
        4. max_words 도달 → 강제 분할
        """
        if not words:
            return []

        SENTENCE_END = set(".!?。！？")
        CLAUSE_BREAK = set(",;:，；：")

        chunks = []
        current_chunk = []

        for i, word in enumerate(words):
            current_chunk.append(word)
            text = word.get("word", "").strip()
            n = len(current_chunk)

            # 다음 단어와의 pause 계산
            has_pause = False
            if i + 1 < len(words):
                gap = words[i + 1].get("start", 0) - word.get("end", 0)
                has_pause = gap >= pause_threshold

            # 1. 문장 종결 → 즉시 분할 (1단어라도 문장 완성이면 OK)
            if text and text[-1] in SENTENCE_END:
                chunks.append(current_chunk)
                current_chunk = []
                continue

            # 2. 절 구분 + pause → 분할
            if n >= min_words and text and text[-1] in CLAUSE_BREAK and has_pause:
                chunks.append(current_chunk)
                current_chunk = []
                continue

            # 3. 단독 pause → 분할
            if n >= min_words and has_pause:
                chunks.append(current_chunk)
                current_chunk = []
                continue

            # 4. max_words 도달 → 강제 분할
            if n >= max_words:
                chunks.append(current_chunk)
                current_chunk = []
                continue

        if current_chunk:
            # 너무 짧은 잔여분은 이전 청크에 병합
            if len(current_chunk) < min_words and chunks:
                chunks[-1].extend(current_chunk)
            else:
                chunks.append(current_chunk)

        return chunks

    def _write_srt(self, segments: list, output_path: Path):
        """segments를 SRT 파일로 저장."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            idx = 1
            for seg in segments:
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                text = seg.get("text", "").strip()
                if not text:
                    continue

                f.write(f"{idx}\n")
                f.write(f"{self._format_timestamp(start)} --> {self._format_timestamp(end)}\n")
                f.write(f"{text}\n\n")
                idx += 1

    def _format_timestamp(self, seconds: float) -> str:
        """초를 SRT 타임스탬프 형식으로 변환 (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")
