"""
Subtitle translator module.
Translates SRT subtitle entries to Korean using Gemini API batch translation.
"""

import json
import logging
import time
from typing import Optional

from .config import (
    GEMINI_API_KEY,
    API_MAX_RETRIES,
    API_RETRY_DELAY,
    TRANSLATION_SETTINGS,
)

logger = logging.getLogger(__name__)


class SubtitleTranslator:
    """Gemini API로 SRT 자막을 한국어로 번역."""

    def __init__(self, model: Optional[str] = None):
        self.model = model or TRANSLATION_SETTINGS["model"]
        self.target_lang = TRANSLATION_SETTINGS["target_lang"]
        self.skip_langs = TRANSLATION_SETTINGS["skip_langs"]

    def should_translate(self, source_lang: str) -> bool:
        """번역이 필요한 언어인지 확인."""
        return source_lang not in self.skip_langs

    def translate_entries(
        self,
        entries: list[dict],
        source_lang: str,
    ) -> list[dict]:
        """
        SRT 항목 리스트를 한국어로 번역.

        Args:
            entries: [{"start": float, "end": float, "text": str}, ...]
            source_lang: 원본 언어 코드 ("en", "ja", ...)

        Returns:
            [{"start": float, "end": float, "original": str, "translated": str}, ...]
        """
        if not entries:
            return []

        if not self.should_translate(source_lang):
            logger.info(f"언어 '{source_lang}' → 번역 스킵")
            return [
                {
                    "start": e["start"],
                    "end": e["end"],
                    "original": e["text"],
                    "translated": "",
                }
                for e in entries
            ]

        logger.info(f"번역 시작: {len(entries)}개 항목 ({source_lang} → {self.target_lang})")

        translated = self._batch_translate(entries, source_lang)
        return translated

    def _batch_translate(
        self,
        entries: list[dict],
        source_lang: str,
    ) -> list[dict]:
        """Gemini API 단일 호출로 배치 번역."""
        from google import genai

        if not GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY 미설정; 번역 스킵")
            return self._fallback_no_translation(entries)

        # 번역 요청 JSON 구성
        texts = [{"id": i, "text": e["text"]} for i, e in enumerate(entries)]

        prompt = f"""You are a subtitle translator. Translate the following subtitle lines from {source_lang} to Korean (한국어).

Rules:
- Keep translations short and natural for subtitles (숏폼 자막용)
- Use casual/conversational Korean (구어체)
- Do NOT add honorifics unless the original clearly uses formal speech
- Preserve the meaning, not word-for-word translation
- Each line should be concise (aim for similar length to original)

Input (JSON array):
{json.dumps(texts, ensure_ascii=False)}

Return ONLY a JSON array with the same structure, replacing "text" with the Korean translation.
Example: [{{"id": 0, "text": "번역된 텍스트"}}]"""

        client = genai.Client(api_key=GEMINI_API_KEY)

        for attempt in range(API_MAX_RETRIES):
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "temperature": 0.3,
                        "max_output_tokens": 8192,
                    },
                )

                if not response.candidates or not response.candidates[0].content:
                    logger.warning(f"Gemini 빈 응답 (attempt {attempt + 1})")
                    if attempt < API_MAX_RETRIES - 1:
                        time.sleep(API_RETRY_DELAY * (2 ** attempt))
                        continue
                    return self._fallback_no_translation(entries)

                text = response.candidates[0].content.parts[0].text.strip()
                text = self._clean_json(text)
                translations = json.loads(text)

                # id → 번역 매핑
                trans_map = {t["id"]: t["text"] for t in translations}

                result = []
                for i, entry in enumerate(entries):
                    result.append({
                        "start": entry["start"],
                        "end": entry["end"],
                        "original": entry["text"],
                        "translated": trans_map.get(i, entry["text"]),
                    })

                logger.info(f"번역 완료: {len(result)}개 항목")
                return result

            except json.JSONDecodeError as e:
                logger.warning(f"번역 JSON 파싱 실패 (attempt {attempt + 1}): {e}")
                if attempt < API_MAX_RETRIES - 1:
                    time.sleep(API_RETRY_DELAY * (2 ** attempt))
            except Exception as e:
                logger.warning(f"번역 API 에러 (attempt {attempt + 1}): {e}")
                if attempt < API_MAX_RETRIES - 1:
                    time.sleep(API_RETRY_DELAY * (2 ** attempt))

        logger.error("번역 실패; 원문만 사용")
        return self._fallback_no_translation(entries)

    def _fallback_no_translation(self, entries: list[dict]) -> list[dict]:
        """번역 실패 시 원문만 반환."""
        return [
            {
                "start": e["start"],
                "end": e["end"],
                "original": e["text"],
                "translated": "",
            }
            for e in entries
        ]

    def _clean_json(self, text: str) -> str:
        """Gemini 응답의 비정규 JSON을 정리."""
        import re

        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
            text = text.strip()
        text = re.sub(r",\s*([}\]])", r"\1", text)
        text = re.sub(r"//[^\n]*", "", text)
        return text.strip()
