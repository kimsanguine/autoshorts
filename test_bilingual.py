"""
이중 자막 테스트: Tim Urban TED 영상의 기존 run에서 subtitle 단계만 재실행.
기존 crop 결과를 재사용하여 영어 원문 + 한국어 번역 이중 자막 생성.
"""

import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("test_bilingual")

# 기존 run 디렉토리
RUN_DIR = Path("temp/run_20260207_102153")
PROGRESS_PATH = RUN_DIR / "progress.json"

with open(PROGRESS_PATH) as f:
    progress = json.load(f)

logger.info(f"Video: {progress['steps']['download']['title']}")
logger.info(f"Language: {progress['steps']['transcribe']['language']}")

# subtitle/finalize 단계 초기화 (재실행용)
progress["steps"].pop("subtitle", None)
progress["steps"].pop("finalize", None)
progress["last_step"] = "crop"

# 이전 output 디렉토리 정리
output_dir = RUN_DIR / "output_bilingual"
if output_dir.exists():
    shutil.rmtree(output_dir)

# progress 임시 저장
with open(PROGRESS_PATH, "w") as f:
    json.dump(progress, f, ensure_ascii=False, indent=2, default=str)

# subtitle 단계 실행
from src.shorts_pipeline.config import TRANSLATION_SETTINGS
from src.shorts_pipeline.transcriber import AudioTranscriber
from src.shorts_pipeline.subtitle_overlay import SubtitleOverlay
from src.shorts_pipeline.subtitle_translator import SubtitleTranslator

# Load data
segments_path = Path(progress["steps"]["analyze"]["segments_path"])
with open(segments_path) as f:
    analysis = json.load(f)
segments = analysis["segments"]

words_path = Path(progress["steps"]["transcribe"]["words_path"])
with open(words_path) as f:
    transcript = json.load(f)
words = transcript["words"]

cropped_paths = [Path(p) for p in progress["steps"]["crop"]["cropped_paths"]]
srt_dir = RUN_DIR / "subtitles_bilingual"
srt_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)

source_lang = progress["steps"]["transcribe"]["language"]  # "en"
logger.info(f"Source language: {source_lang}")
logger.info(f"Translation enabled: {source_lang not in TRANSLATION_SETTINGS['skip_langs']}")

# 번역기 생성
translator = SubtitleTranslator()

# 자막 오버레이 (이중 자막 모드)
overlay = SubtitleOverlay(
    style="outline",
    font_preset="noto_sans_kr",
    translator=translator,
    source_lang=source_lang,
)
logger.info(f"Bilingual mode: {overlay._bilingual}")

# Build cropped_results
cropped_results = []
for seg, cp in zip(segments, cropped_paths):
    cropped_results.append({"segment": seg, "cropped_path": cp})

transcriber = AudioTranscriber()
results = overlay.apply_batch(
    cropped_results=cropped_results,
    transcriber=transcriber,
    words=words,
    output_dir=output_dir,
    srt_dir=srt_dir,
)

logger.info("=" * 50)
logger.info("테스트 완료!")
for r in results:
    logger.info(f"  Output: {r['output_path']}")
logger.info(f"출력 디렉토리: {output_dir}")
