"""
Segment selection algorithm for YouTube-to-Shorts pipeline.
Combines heatmap peaks and Gemini content analysis to find viral-worthy segments.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import find_peaks

from .config import (
    SEGMENT_SETTINGS,
    SIGNAL_WEIGHTS,
    FALLBACK_WEIGHTS,
    GEMINI_API_KEY,
    GEMINI_ANALYSIS_MODEL,
    API_MAX_RETRIES,
    API_RETRY_DELAY,
)

logger = logging.getLogger(__name__)


class SegmentSelector:
    """Heatmap + Gemini 분석으로 바이럴 구간을 선택한다."""

    def __init__(
        self,
        n_shorts: Optional[int] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
    ):
        self.n_shorts = n_shorts or SEGMENT_SETTINGS["n_shorts"]
        self.min_duration = min_duration or SEGMENT_SETTINGS["min_duration"]
        self.max_duration = max_duration or SEGMENT_SETTINGS["max_duration"]
        self.min_gap = SEGMENT_SETTINGS["min_gap"]
        self.nms_iou_threshold = SEGMENT_SETTINGS["nms_iou_threshold"]

    def analyze(
        self,
        heatmap: Optional[list],
        transcript: dict,
        video_duration: float,
        output_dir: Optional[Path] = None,
    ) -> list:
        """
        바이럴 가능성이 높은 구간을 분석하고 선택한다.

        Args:
            heatmap: List of {"start_time", "end_time", "value"} or None
            transcript: {"words": [...], "segments": [...], "text": str}
            video_duration: 전체 영상 길이 (초)
            output_dir: 분석 결과 저장 경로

        Returns:
            List of segment dicts, sorted by score descending:
            [{"start": float, "end": float, "score": float,
              "score_breakdown": {"heatmap": float, "content": float},
              "reason": str}, ...]
        """
        has_heatmap = heatmap is not None and len(heatmap) > 0
        weights = SIGNAL_WEIGHTS if has_heatmap else FALLBACK_WEIGHTS

        # Phase 1에서는 structure 신호 미구현 → heatmap + content만 사용
        # structure 가중치를 content에 합산
        effective_weights = {
            "heatmap": weights["heatmap"],
            "content": weights["content"] + weights["structure"],
        }

        logger.info(
            f"Analyzing segments (heatmap={'yes' if has_heatmap else 'no'}, "
            f"weights: heatmap={effective_weights['heatmap']:.2f}, "
            f"content={effective_weights['content']:.2f})"
        )

        # Step 1: Heatmap scoring
        heatmap_candidates = []
        if has_heatmap:
            heatmap_candidates = self._score_heatmap(heatmap, video_duration)
            logger.info(f"Heatmap candidates: {len(heatmap_candidates)}")

        # Step 2: Content scoring via Gemini
        content_candidates = self._score_content(transcript, video_duration)
        logger.info(f"Content candidates: {len(content_candidates)}")

        # Step 3: Merge and combine scores
        candidates = self._merge_candidates(
            heatmap_candidates, content_candidates, effective_weights, video_duration
        )
        logger.info(f"Merged candidates: {len(candidates)}")

        # Step 4: Non-maximum suppression
        selected = self._nms(candidates)
        logger.info(f"After NMS: {len(selected)}")

        # Step 5: Filter out segments beyond video boundary
        selected = [
            s for s in selected
            if s["start"] < video_duration - self.min_duration
        ]

        # Step 6: Top-N selection
        selected = selected[: self.n_shorts]

        # Step 7: Snap boundaries to natural pauses
        words = transcript.get("words", [])
        selected = [self._snap_boundaries(seg, words) for seg in selected]

        # Clamp end to video duration
        for seg in selected:
            seg["end"] = min(seg["end"], video_duration)

        logger.info(f"Final segments: {len(selected)}")
        for i, seg in enumerate(selected):
            logger.info(
                f"  #{i+1}: {seg['start']:.1f}s - {seg['end']:.1f}s "
                f"(score={seg['score']:.3f}, reason={seg['reason']})"
            )

        # Save results
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            segments_path = output_dir / "segments.json"
            with open(segments_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "n_shorts": self.n_shorts,
                        "has_heatmap": has_heatmap,
                        "weights": effective_weights,
                        "segments": selected,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            logger.info(f"Saved: {segments_path.name}")

        return selected

    # ------------------------------------------------------------------
    # Signal 1: Heatmap
    # ------------------------------------------------------------------

    def _score_heatmap(self, heatmap: list, video_duration: float) -> list:
        """
        Heatmap에서 피크를 찾고, 각 피크 주변 window의 스코어를 계산한다.

        Returns:
            List of {"start", "end", "heatmap_score", "peak_time"}
        """
        values = np.array([entry["value"] for entry in heatmap])
        times = np.array([(entry["start_time"] + entry["end_time"]) / 2 for entry in heatmap])

        # heatmap bin 간격
        bin_duration = heatmap[1]["start_time"] - heatmap[0]["start_time"] if len(heatmap) > 1 else 8.0

        # minimum distance between peaks (in bins)
        min_distance = max(1, int(self.min_gap / bin_duration))

        # Find peaks
        peaks, properties = find_peaks(
            values,
            prominence=0.10,
            distance=min_distance,
            height=np.percentile(values, 30),  # 하위 30% 이하는 무시
        )

        if len(peaks) == 0:
            # No prominent peaks → top-K highest values
            logger.warning("No prominent peaks found; using top values")
            top_indices = np.argsort(values)[::-1][: self.n_shorts * 2]
            peaks = top_indices

        candidates = []
        target_duration = (self.min_duration + self.max_duration) / 2

        for peak_idx in peaks:
            peak_time = times[peak_idx]

            # window 중심을 peak에 맞추되, 영상 범위 내로 clamp
            half = target_duration / 2
            start = max(0, peak_time - half)
            end = min(video_duration, peak_time + half)

            # 최소 길이 보장
            if end - start < self.min_duration:
                if start == 0:
                    end = min(video_duration, self.min_duration)
                else:
                    start = max(0, end - self.min_duration)

            # window 내 heatmap 평균값
            window_mask = (times >= start) & (times <= end)
            window_values = values[window_mask]
            avg_value = float(np.mean(window_values)) if len(window_values) > 0 else float(values[peak_idx])

            # Normalize: peak의 prominence도 반영
            peak_value = float(values[peak_idx])
            score = 0.6 * avg_value + 0.4 * peak_value

            candidates.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "heatmap_score": round(score, 4),
                "peak_time": round(peak_time, 2),
            })

        # Sort by score descending
        candidates.sort(key=lambda x: x["heatmap_score"], reverse=True)
        return candidates

    # ------------------------------------------------------------------
    # Signal 2: Content Analysis (Gemini)
    # ------------------------------------------------------------------

    def _score_content(self, transcript: dict, video_duration: float) -> list:
        """
        Gemini API로 트랜스크립트를 분석하여 바이럴 구간을 식별한다.

        Returns:
            List of {"start", "end", "content_score", "reason"}
        """
        from google import genai

        if not GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not set; skipping content analysis")
            return []

        # 트랜스크립트를 타임스탬프 포함 텍스트로 변환
        segments = transcript.get("segments", [])
        if not segments:
            logger.warning("No segments in transcript; skipping content analysis")
            return []

        timestamped_text = self._format_transcript_for_gemini(segments)

        prompt = f"""You are a YouTube Shorts/Reels viral content expert.

Below is a YouTube video transcript with timestamps.
Select the {self.n_shorts * 2} best segments (each 15-20 seconds) for creating viral short-form clips.

Selection criteria:
1. Hook strength: Does the first 3 seconds grab attention?
2. Emotional impact: Does it evoke surprise, empathy, humor, or insight?
3. Self-contained: Does it deliver a complete message in 15-20 seconds?
4. Quotability: Does it contain shareable quotes or statements?
5. Engagement: Would it drive comments and discussion?

Transcript:
{timestamped_text}

Return ONLY a JSON array. Each element must have exactly these 4 keys:
- "start": start time in seconds (number)
- "end": end time in seconds (number)
- "score": viral potential 0.0-1.0 (number)
- "reason": one-line explanation (string)
"""

        client = genai.Client(api_key=GEMINI_API_KEY)

        for attempt in range(API_MAX_RETRIES):
            try:
                response = client.models.generate_content(
                    model=GEMINI_ANALYSIS_MODEL,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "temperature": 0.3,
                        "max_output_tokens": 8192,
                    },
                )

                # Null-check (Gemini 응답 안전 처리)
                if not response.candidates or not response.candidates[0].content:
                    logger.warning(f"Gemini returned empty response (attempt {attempt+1})")
                    if attempt < API_MAX_RETRIES - 1:
                        time.sleep(API_RETRY_DELAY * (2 ** attempt))
                        continue
                    return []

                text = response.candidates[0].content.parts[0].text.strip()
                text = self._clean_json(text)
                logger.info(f"Gemini response length: {len(text)} chars")
                raw_segments = json.loads(text)
                logger.info(f"Parsed {len(raw_segments)} raw segments from Gemini")

                # Validate and normalize
                candidates = []
                for seg in raw_segments:
                    start = float(seg.get("start", 0))
                    end = float(seg.get("end", 0))
                    score = float(seg.get("score", 0.5))
                    reason = seg.get("reason", "")

                    # 범위 검증
                    if end <= start:
                        continue
                    duration = end - start
                    if duration < self.min_duration * 0.5:  # 너무 짧은 건 무시
                        continue
                    if duration > self.max_duration * 2:  # 너무 긴 건 잘라냄
                        end = start + self.max_duration

                    # Clamp to video
                    start = max(0, min(start, video_duration))
                    end = max(start + 1, min(end, video_duration))

                    candidates.append({
                        "start": round(start, 2),
                        "end": round(end, 2),
                        "content_score": round(min(1.0, max(0.0, score)), 4),
                        "reason": reason,
                    })

                candidates.sort(key=lambda x: x["content_score"], reverse=True)
                return candidates

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse Gemini JSON (attempt {attempt+1}): {e}")
                logger.debug(f"Raw Gemini response:\n{text[:500]}")
                if attempt < API_MAX_RETRIES - 1:
                    time.sleep(API_RETRY_DELAY * (2 ** attempt))
            except Exception as e:
                logger.warning(f"Gemini API error (attempt {attempt+1}): {e}")
                if attempt < API_MAX_RETRIES - 1:
                    time.sleep(API_RETRY_DELAY * (2 ** attempt))

        logger.error("All Gemini API attempts failed; returning empty content candidates")
        return []

    def _clean_json(self, text: str) -> str:
        """Gemini 응답의 비정규 JSON을 정리한다."""
        # 마크다운 코드블록 제거
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
            text = text.strip()
        # Trailing commas 제거: },] → }] / ,] → ] / ,} → }
        text = re.sub(r",\s*([}\]])", r"\1", text)
        # JavaScript-style comments 제거
        text = re.sub(r"//[^\n]*", "", text)
        # NaN, Infinity → null
        text = re.sub(r"\bNaN\b", "null", text)
        text = re.sub(r"\bInfinity\b", "null", text)
        return text.strip()

    def _format_transcript_for_gemini(self, segments: list) -> str:
        """segments를 Gemini에 보낼 타임스탬프 포함 텍스트로 변환."""
        lines = []
        for seg in segments:
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            text = seg.get("text", "").strip()
            if text:
                minutes_s = int(start // 60)
                seconds_s = int(start % 60)
                minutes_e = int(end // 60)
                seconds_e = int(end % 60)
                lines.append(
                    f"[{minutes_s:02d}:{seconds_s:02d} - {minutes_e:02d}:{seconds_e:02d}] {text}"
                )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Merge & NMS
    # ------------------------------------------------------------------

    def _merge_candidates(
        self,
        heatmap_candidates: list,
        content_candidates: list,
        weights: dict,
        video_duration: float,
    ) -> list:
        """
        Heatmap과 Content 후보를 병합하고 가중 합산 스코어를 계산한다.

        겹치는 구간은 병합하고, 겹치지 않는 구간은 단일 신호 스코어로 유지.
        """
        all_candidates = []

        # Heatmap candidates에 content score 매칭
        for hc in heatmap_candidates:
            best_content = self._find_best_overlap(hc, content_candidates)
            content_score = best_content["content_score"] if best_content else 0.0
            reason = best_content["reason"] if best_content else "Heatmap peak"

            combined = weights["heatmap"] * hc["heatmap_score"] + weights["content"] * content_score
            all_candidates.append({
                "start": hc["start"],
                "end": hc["end"],
                "score": round(combined, 4),
                "score_breakdown": {
                    "heatmap": round(hc["heatmap_score"], 4),
                    "content": round(content_score, 4),
                },
                "reason": reason,
            })

        # Content candidates 중 heatmap과 겹치지 않는 것 추가
        for cc in content_candidates:
            if not self._has_overlap(cc, heatmap_candidates):
                heatmap_score = 0.0
                combined = weights["heatmap"] * heatmap_score + weights["content"] * cc["content_score"]
                all_candidates.append({
                    "start": cc["start"],
                    "end": cc["end"],
                    "score": round(combined, 4),
                    "score_breakdown": {
                        "heatmap": 0.0,
                        "content": round(cc["content_score"], 4),
                    },
                    "reason": cc["reason"],
                })

        # Sort by combined score
        all_candidates.sort(key=lambda x: x["score"], reverse=True)
        return all_candidates

    def _find_best_overlap(self, target: dict, candidates: list) -> Optional[dict]:
        """target과 가장 많이 겹치는 candidate를 찾는다."""
        best = None
        best_iou = 0

        for c in candidates:
            iou = self._compute_iou(target, c)
            if iou > 0.1 and iou > best_iou:  # 최소 10% 겹침
                best = c
                best_iou = iou

        return best

    def _has_overlap(self, target: dict, candidates: list, threshold: float = 0.1) -> bool:
        """target이 candidates 중 하나와 threshold 이상 겹치는지 확인."""
        for c in candidates:
            if self._compute_iou(target, c) > threshold:
                return True
        return False

    def _compute_iou(self, a: dict, b: dict) -> float:
        """두 구간의 IoU(Intersection over Union)를 계산."""
        start_a, end_a = a["start"], a["end"]
        start_b, end_b = b["start"], b["end"]

        intersection_start = max(start_a, start_b)
        intersection_end = min(end_a, end_b)
        intersection = max(0, intersection_end - intersection_start)

        union = (end_a - start_a) + (end_b - start_b) - intersection
        if union <= 0:
            return 0.0
        return intersection / union

    def _nms(self, candidates: list) -> list:
        """Non-maximum suppression으로 겹치는 구간을 제거한다."""
        if not candidates:
            return []

        # Already sorted by score descending
        selected = []
        for candidate in candidates:
            # 이미 선택된 구간과 겹치는지 확인
            overlaps = False
            for s in selected:
                if self._compute_iou(candidate, s) > self.nms_iou_threshold:
                    overlaps = True
                    break
                # 최소 간격 확인
                gap = max(candidate["start"] - s["end"], s["start"] - candidate["end"])
                if gap < self.min_gap:
                    overlaps = True
                    break

            if not overlaps:
                selected.append(candidate)

        return selected

    # ------------------------------------------------------------------
    # Boundary snapping
    # ------------------------------------------------------------------

    def _snap_boundaries(self, segment: dict, words: list) -> dict:
        """구간의 시작/끝을 단어 경계(pause)에 맞춘다."""
        if not words:
            return segment

        start = segment["start"]
        end = segment["end"]

        # 시작점: 가장 가까운 단어 gap(0.3초 이상) 찾기
        start = self._snap_to_pause(start, words, direction="backward", search_range=2.0)

        # 끝점: 가장 가까운 단어 gap 찾기
        end = self._snap_to_pause(end, words, direction="forward", search_range=2.0)

        # 길이 제한 적용
        duration = end - start
        if duration < self.min_duration:
            end = start + self.min_duration
        elif duration > self.max_duration:
            end = start + self.max_duration

        segment = dict(segment)
        segment["start"] = round(start, 2)
        segment["end"] = round(end, 2)
        return segment

    def _snap_to_pause(
        self, time_point: float, words: list, direction: str = "backward", search_range: float = 2.0
    ) -> float:
        """
        time_point 근처의 pause(단어 간 gap)를 찾아 snap한다.

        Args:
            time_point: 기준 시간
            direction: "backward" (이전 pause) / "forward" (다음 pause)
            search_range: 검색 범위 (초)
        """
        # time_point 근처 단어들 찾기
        nearby_words = [
            w for w in words
            if abs(w.get("start", 0) - time_point) < search_range
            or abs(w.get("end", 0) - time_point) < search_range
        ]

        if len(nearby_words) < 2:
            return time_point

        # 단어 간 gap 찾기
        nearby_words.sort(key=lambda w: w.get("start", 0))
        best_gap_time = time_point
        best_gap_size = 0

        for i in range(len(nearby_words) - 1):
            gap_start = nearby_words[i].get("end", 0)
            gap_end = nearby_words[i + 1].get("start", 0)
            gap_size = gap_end - gap_start

            if gap_size < 0.2:  # 너무 작은 gap은 무시
                continue

            gap_mid = (gap_start + gap_end) / 2

            if direction == "backward" and gap_mid <= time_point and gap_size > best_gap_size:
                best_gap_time = gap_mid
                best_gap_size = gap_size
            elif direction == "forward" and gap_mid >= time_point and gap_size > best_gap_size:
                best_gap_time = gap_mid
                best_gap_size = gap_size

        return best_gap_time
