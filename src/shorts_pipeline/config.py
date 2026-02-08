"""
Configuration module for YouTube-to-Shorts pipeline.
Manages API keys, download settings, segment analysis, and video composition.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# === Base Paths ===
PROJECT_ROOT = Path(__file__).parent.parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUT_DIR = PROJECT_ROOT / "output"
TEMP_DIR = PROJECT_ROOT / "temp"

# Create directories
for d in [ASSETS_DIR, OUTPUT_DIR, TEMP_DIR, ASSETS_DIR / "font"]:
    d.mkdir(exist_ok=True)

# === API Keys ===
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")  # Optional

# === API Retry ===
API_MAX_RETRIES = 3
API_RETRY_DELAY = 2  # seconds, exponential backoff (2s, 4s, 8s)

# === Download Settings ===
DOWNLOAD_SETTINGS = {
    "max_resolution": "1080p",
    "preferred_format": "mp4",
    "extract_heatmap": True,
    "extract_auto_subs": True,
    "sub_languages": ["ko", "en"],
}

# === Transcription Settings ===
TRANSCRIPTION_SETTINGS = {
    "model": "large-v3",
    "language": "ko",
    "device": "cpu",             # CTranslate2(faster-whisper)는 MPS 미지원, CPU 사용
    "compute_type": "int8",      # CPU에서는 int8이 최적
    "batch_size": 16,
}

# === Segment Selection ===
SEGMENT_SETTINGS = {
    "n_shorts": 5,              # 생성할 숏폼 개수
    "min_duration": 15,         # 최소 길이 (초)
    "max_duration": 20,         # 최대 길이 (초)
    "min_gap": 10,              # 구간 간 최소 간격 (초)
    "nms_iou_threshold": 0.3,   # Non-max suppression overlap 임계값
}

SIGNAL_WEIGHTS = {
    "heatmap": 0.45,
    "content": 0.35,
    "structure": 0.20,
}

FALLBACK_WEIGHTS = {
    "heatmap": 0.0,
    "content": 0.60,
    "structure": 0.40,
}

GEMINI_ANALYSIS_MODEL = "gemini-2.5-flash"

# === Translation Settings ===
TRANSLATION_SETTINGS = {
    "enabled": True,               # 번역 활성화
    "target_lang": "ko",           # 번역 대상 언어
    "skip_langs": ["ko"],          # 이 언어면 번역 스킵
    "model": "gemini-2.5-flash",   # 번역 모델
}

# === Bilingual Subtitle Settings ===
BILINGUAL_SETTINGS = {
    "original_fontsize_ratio": 0.75,  # 원문: 기본 fontsize의 75%
    "translated_fontsize_ratio": 1.0, # 번역: 기본 fontsize 100%
    "line_gap": 8,                    # 원문-번역 간격 (px)
}

# === Video Crop ===
CROP_MODE = "center"  # "center" or "blurred_bg"

# === Font Presets ===
# 10종 폰트 프리셋 (번들 7개 + 시스템 3개)
FONT_DIR = ASSETS_DIR / "font"

FONT_PRESETS = {
    "kotra_songeulssi": {
        "path": str(FONT_DIR / "KOTRA_SONGEULSSI.otf"),
        "name": "KOTRA_SONGEULSSI",
        "mood": "casual",
        "description": "캐주얼/감성 손글씨",
    },
    "sd_gothic": {
        "path": "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "name": "Apple SD Gothic Neo",
        "mood": "clean",
        "description": "깔끔/기본 고딕",
    },
    "noto_sans_kr": {
        "path": str(FONT_DIR / "NotoSansKR-Bold.ttf"),
        "name": "Noto Sans KR",
        "mood": "modern",
        "description": "모던/범용 고딕",
    },
    "black_han_sans": {
        "path": str(FONT_DIR / "BlackHanSans-Regular.ttf"),
        "name": "Black Han Sans",
        "mood": "impact",
        "description": "굵고 임팩트 있는",
    },
    "do_hyeon": {
        "path": str(FONT_DIR / "DoHyeon-Regular.ttf"),
        "name": "Do Hyeon",
        "mood": "pop",
        "description": "둥글고 팝한 느낌",
    },
    "jua": {
        "path": str(FONT_DIR / "Jua-Regular.ttf"),
        "name": "Jua",
        "mood": "cute",
        "description": "귀엽고 동글동글",
    },
    "nanum_gothic": {
        "path": str(FONT_DIR / "NanumGothic-Bold.ttf"),
        "name": "NanumGothic",
        "mood": "classic",
        "description": "클래식 고딕",
    },
    "nanum_pen": {
        "path": str(FONT_DIR / "NanumPenScript-Regular.ttf"),
        "name": "Nanum Pen",
        "mood": "handwriting",
        "description": "손글씨/자유로운",
    },
    "gugi": {
        "path": str(FONT_DIR / "Gugi-Regular.ttf"),
        "name": "Gugi",
        "mood": "trendy",
        "description": "트렌디/마케팅",
    },
    "ibm_plex_sans_kr": {
        "path": str(FONT_DIR / "IBMPlexSansKR-Bold.ttf"),
        "name": "IBM Plex Sans KR",
        "mood": "technical",
        "description": "기술/정보 전달",
    },
}

DEFAULT_FONT_PRESET = "noto_sans_kr"

# 분위기 → 폰트 매핑 (Gemini 분석 결과 → 프리셋 선택)
MOOD_TO_FONT = {
    "casual": "kotra_songeulssi",
    "clean": "sd_gothic",
    "modern": "noto_sans_kr",
    "impact": "black_han_sans",
    "pop": "do_hyeon",
    "cute": "jua",
    "classic": "nanum_gothic",
    "handwriting": "nanum_pen",
    "trendy": "gugi",
    "technical": "ibm_plex_sans_kr",
}

# === Subtitle Styles ===
# 3가지 자막 스타일
SUBTITLE_STYLES = {
    "box": {
        "description": "반투명 검정 배경 (기본, 어떤 영상에서도 가독성 보장)",
        # ASS subtitles filter 파라미터
        "ass_style": {
            "BorderStyle": 4,        # 배경 박스
            "BackColour": "&H80000000",  # 반투명 검정
            "Outline": 0,
            "Shadow": 0,
        },
        # drawtext fallback 파라미터
        "drawtext_style": {
            "box": 1,
            "boxcolor": "black@0.5",
            "boxborderw": 8,
            "borderw": 0,
        },
    },
    "outline": {
        "description": "테두리+그림자 (깔끔, 밝은 장면에서 약간 읽기 어려울 수 있음)",
        "ass_style": {
            "BorderStyle": 1,
            "Outline": 3,
            "Shadow": 2,
            "OutlineColour": "&H00000000",
        },
        "drawtext_style": {
            "borderw": 3,
            "bordercolor": "black",
            "shadowx": 2,
            "shadowy": 2,
            "shadowcolor": "black",
        },
    },
    "highlight": {
        "description": "현재 단어 강조 (word-level timestamps 활용, 노란색 하이라이트)",
        "ass_style": {
            "BorderStyle": 1,
            "Outline": 2,
            "Shadow": 1,
            "OutlineColour": "&H00000000",
        },
        "drawtext_style": {
            "borderw": 2,
            "bordercolor": "black",
        },
        "highlight_color": "yellow",
    },
}

DEFAULT_SUBTITLE_STYLE = "outline"

# === Subtitle Position ===
SUBTITLE_POSITION_POLICY = {
    "shorts": {"bottom_pct": 0.20},  # 20% from bottom (UI 겹침 방지)
}

# === Video Settings ===
VIDEO_SETTINGS = {
    "resolution": (1080, 1920),     # 9:16 세로
    "fps": 30,
    "fade_duration": 0.3,
    "subtitle_font": str(FONT_DIR / "NotoSansKR-Bold.ttf"),
    "subtitle_fontsize": 58,        # 모바일 가독성 강화 (52→58)
    "subtitle_color": "white",
    "subtitle_margin": int(1920 * 0.20),  # 384px (25%→20%, Shorts 하단 UI 겹침 방지)
    "video_bitrate": "8M",
    "audio_bitrate": "192k",
}


# === Helper Functions ===
def get_timestamp() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_output_path(filename: str) -> Path:
    return OUTPUT_DIR / filename


def get_temp_path(filename: str) -> Path:
    return TEMP_DIR / filename
