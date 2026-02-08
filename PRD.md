# PRD: `260207_shortform_agent` — YouTube-to-Shorts 자동 변환 파이프라인

## Context

`260203_longform_agent`는 명언 텍스트에서 영상을 생성하는 파이프라인이다. 이번 프로젝트는 **반대 방향** — 이미 존재하는 YouTube 영상에서 바이럴 가능성이 높은 구간을 자동 식별하여 15-20초 숏폼 여러 개를 생성한다. longform_agent의 파이프라인 패턴(config, progress resume, retry, FFmpeg 합성)을 그대로 재사용한다.

---

## 1. 제품 개요

**Input**: YouTube 영상 URL 1개
**Output**: 9:16 세로 숏폼 영상 3~8개 (각 15-20초, 이중 자막 포함)

**핵심 가치**: YouTube "Most Replayed" heatmap + Gemini 트랜스크립트 분석의 2-track 접근으로 가장 바이럴 가능성 높은 구간을 자동 선별

---

## 2. 핵심 파이프라인

```
[Step 1] Download & Extract
  yt-dlp → 영상 + 메타데이터 + heatmap + 자동자막
     ↓
[Step 2] Transcribe
  WhisperX → word-level timestamps (device=cpu, compute_type=int8)
     ↓
[Step 3] Analyze & Select Segments  ← 핵심 모듈
  Heatmap peaks + Gemini 분석 → 스코어링 → Top-N 선택
     ↓
[Step 4] Extract & Crop
  FFmpeg → 구간 추출 + 16:9→9:16 세로 변환
     ↓
[Step 5] Subtitle Overlay + Translation
  ┌─ 원본 언어 == skip_langs(ko)? → 단일 자막 (한국어)
  └─ 원본 언어 != skip_langs?    → Gemini 배치 번역 → 이중 자막 (원문+한국어)
  WhisperX words → 자연어 word grouping → drawtext/subtitles filter
     ↓
[Step 6] Finalize & Output
  출력 디렉토리 복사 → metadata.json 생성
```

---

## 3. 세그먼트 선택 알고리즘 (핵심)

3가지 신호를 결합한 가중 스코어링:

| Signal | Weight | Source | Fallback (heatmap 없을 때) |
|--------|--------|--------|---------------------------|
| Heatmap | 0.45 | yt-dlp "Most Replayed" 데이터 | 0.0 |
| Content | 0.35 | Gemini 트랜스크립트 분석 | 0.60 |
| Structure | 0.20 | 미구현 → content에 합산 | 0.40 |

> **현재 구현**: Structure 신호는 미구현. `effective_weights`에서 structure 가중치를 content에 합산하여 2-signal로 운영 (heatmap 있을 때: heatmap=0.45, content=0.55 / 없을 때: content=1.0).

### Signal 1: Heatmap Analysis
- yt-dlp에서 추출한 heatmap values → `scipy.signal.find_peaks(prominence=0.10, distance=동적)`
  - `distance`: `min_gap / bin_duration`으로 동적 계산
- 각 peak 주변 15-20초 window의 평균 value를 스코어로
- heatmap 없는 영상(조회수 <50K)은 가중치 0으로, 나머지 재분배

### Signal 2: Content Analysis
- Gemini API (`gemini-2.5-flash`)에 전체 트랜스크립트 전달
- 평가: 감정적 임팩트, 자기완결성(15-20초 내 의미 전달), Hook 강도, 인용 가능성
- 출력: `[{start_time, end_time, score, reason}]`

### Signal 3: Structure Analysis (미구현)
- Phase 2 예정: PySceneDetect + librosa + pydub
- 현재는 structure 가중치가 content에 합산됨

### Selection Pipeline
1. 전체 영상을 5초 sliding window (3초 overlap)로 스캔
2. 각 window에 heatmap + content score 부여
3. 가중 합산 → Non-maximum suppression (IoU > 0.3 중복 제거)
4. Top-N 선택 → 시작/끝을 silence/scene boundary에 snap
5. 최종 길이 15-20초로 조정

---

## 4. 기술 아키텍처

### Pipeline 클래스

```python
class ShortsPipeline:
    STEPS = ["download", "transcribe", "analyze", "crop", "subtitle", "finalize"]

    def __init__(
        self,
        n_shorts: int = None,       # default: 5
        crop_mode: str = None,      # "center" (default) | "blurred_bg"
        enable_subtitles: bool = True,
        subtitle_style: str = None, # "box" | "outline" (default) | "highlight"
        font_preset: str = None,    # "noto_sans_kr" (default), 10종 중 택1
    ): ...

    def generate(
        self,
        youtube_url: str,
        output_dir: Optional[Path] = None,
        keep_temp_files: bool = True,
        resume_from: Optional[Path] = None,
    ) -> dict: ...  # {"output_paths": [...], "metadata": {...}, "run_dir": Path}
```

`progress.json` 기반 resume — longform_agent 패턴 동일.

### Config 핵심 설정

```python
DOWNLOAD_SETTINGS = {
    "max_resolution": "1080p",
    "preferred_format": "mp4",
    "extract_heatmap": True,
    "extract_auto_subs": True,
    "sub_languages": ["ko", "en"],
}

TRANSCRIPTION_SETTINGS = {
    "model": "large-v3",
    "language": "ko",
    "device": "cpu",             # CTranslate2(faster-whisper)는 MPS 미지원
    "compute_type": "int8",      # CPU에서는 int8이 최적
    "batch_size": 16,
}

SEGMENT_SETTINGS = {
    "n_shorts": 5,
    "min_duration": 15,
    "max_duration": 20,
    "min_gap": 10,
    "nms_iou_threshold": 0.3,
}

SIGNAL_WEIGHTS = {"heatmap": 0.45, "content": 0.35, "structure": 0.20}
FALLBACK_WEIGHTS = {"heatmap": 0.0, "content": 0.60, "structure": 0.40}

TRANSLATION_SETTINGS = {
    "enabled": True,
    "target_lang": "ko",
    "skip_langs": ["ko"],        # 한국어 영상은 번역 스킵
    "model": "gemini-2.5-flash",
}

BILINGUAL_SETTINGS = {
    "original_fontsize_ratio": 0.75,  # 원문: 기본 fontsize의 75%
    "translated_fontsize_ratio": 1.0, # 번역: 기본 fontsize 100%
    "line_gap": 8,                    # 원문-번역 간격 (px)
}

VIDEO_SETTINGS = {
    "resolution": (1080, 1920),
    "fps": 30,
    "fade_duration": 0.3,
    "subtitle_font": "NotoSansKR-Bold.ttf",
    "subtitle_fontsize": 58,                  # 모바일 가독성 강화
    "subtitle_color": "white",
    "subtitle_margin": int(1920 * 0.20),      # 384px (하단 20%)
    "video_bitrate": "8M",
    "audio_bitrate": "192k",
}
```

---

## 5. 입출력 포맷

### Input
```python
pipeline.generate("https://www.youtube.com/watch?v=VIDEO_ID")
```
지원 형식: `youtube.com/watch?v=`, `youtu.be/`, `youtube.com/shorts/` (숏폼이면 경고)

### Output
```
output/
├── VIDEO_ID_short_001.mp4          # 1080x1920, 15-20초
├── VIDEO_ID_short_002.mp4
├── ...
└── VIDEO_ID_metadata.json          # 생성 메타데이터 + 스코어 breakdown
```

`metadata.json`:
```json
{
  "source": {"url": "...", "title": "...", "duration": 1234.5, "has_heatmap": true},
  "shorts": [
    {
      "index": 0, "file": "short_001.mp4",
      "source_start": 125.3, "source_end": 141.8,
      "score": 0.87,
      "score_breakdown": {"heatmap": 0.92, "content": 0.84},
      "reason": "감정적 임팩트 + 높은 heatmap peak"
    }
  ]
}
```

**영상 스펙**: 1080x1920 (9:16), H.264, 30fps, AAC 192kbps, 8Mbps, 자막 burn-in

---

## 6. 자막 시스템

### 6.1 폰트 프리셋 (10종)

| 프리셋 키 | 폰트 | 분위기 | 소스 |
|-----------|------|--------|------|
| `kotra_songeulssi` | KOTRA_SONGEULSSI.otf | casual | bundled |
| `sd_gothic` | Apple SD Gothic Neo | clean | system |
| `noto_sans_kr` (기본) | NotoSansKR-Bold.ttf | modern | bundled |
| `black_han_sans` | BlackHanSans-Regular.ttf | impact | bundled |
| `do_hyeon` | DoHyeon-Regular.ttf | pop | bundled |
| `jua` | Jua-Regular.ttf | cute | bundled |
| `nanum_gothic` | NanumGothic-Bold.ttf | classic | bundled |
| `nanum_pen` | NanumPenScript-Regular.ttf | handwriting | bundled |
| `gugi` | Gugi-Regular.ttf | trendy | bundled |
| `ibm_plex_sans_kr` | IBMPlexSansKR-Bold.ttf | technical | bundled |

Gemini 분석 결과의 분위기(mood) → `MOOD_TO_FONT` 매핑으로 자동 선택.

### 6.2 자막 스타일 (3종)

| 스타일 | 설명 | 용도 |
|--------|------|------|
| `box` | 반투명 검정 배경 | 어떤 영상에서도 가독성 보장 |
| `outline` (기본) | 테두리+그림자 | 깔끔, 일반적 |
| `highlight` | 단어 하이라이트 | word-level timestamps 활용 |

### 6.3 이중 자막 (Bilingual Subtitles)

원본 언어가 `skip_langs`(ko)가 아닌 경우 자동 활성화:
- **원문 (위)**: fontsize 43px (58 × 0.75), 85% opacity
- **한국어 번역 (아래)**: fontsize 58px (100%), 100% opacity
- 간격: 8px (`line_gap`)

번역: `SubtitleTranslator` — Gemini 2.5 Flash 배치 번역 (SRT 항목 일괄 전송)
Fallback: API 실패 시 원문만 표시 (단일 자막)

### 6.4 Word Grouping

자연어 기반 단어 분할로 자막 가독성 향상:
1. 문장 종결 (`.!?`) → 무조건 분할
2. 절 구분 + pause → 분할
3. pause만 → 분할
4. max_words 초과 → 강제 분할

### 6.5 Fallback 체인

```
subtitles filter (libass) → drawtext filter → copy (자막 없이)
```

FFmpeg path escape: `replace("\\", "/").replace(":", "\\:")`

---

## 7. MVP 모듈

| 모듈 | 기능 |
|------|------|
| `downloader.py` | yt-dlp 다운로드 + heatmap + info.json 파싱 |
| `transcriber.py` | WhisperX word-level transcription (CPU, int8) |
| `segment_selector.py` | Heatmap peak + Gemini content analysis (2-signal) |
| `video_cropper.py` | FFmpeg center crop (16:9→9:16) |
| `subtitle_overlay.py` | 자막 burn-in (3종 스타일, 이중 자막, fallback 체인) |
| `subtitle_translator.py` | Gemini 배치 번역 (SRT → 한국어) |
| `pipeline.py` | 전체 오케스트레이션 + progress resume |
| `config.py` | 중앙 설정 (경로, API, 폰트, 자막, 번역) |

### Phase 2
- Structure analysis 신호 추가 (PySceneDetect + librosa)
- Blurred background crop 모드
- 얼굴 인식 smart crop (MediaPipe)
- Hook 텍스트 자동 생성 (Gemini)
- 병렬 처리 (concurrent.futures)

### Phase 3
- YouTube Data API 연동 (댓글 타임스탬프 파싱)
- Speaker diarization → 화자별 highlight
- 배치 처리 (여러 URL 동시)
- Claude Code agent/skill 완성

---

## 8. Dependencies

### requirements.txt
```
yt-dlp>=2024.1.0              # YouTube 다운로드 + heatmap
google-genai>=1.0.0           # Gemini API (트랜스크립트 분석 + 번역)
ffmpeg-python>=0.2.0          # FFmpeg wrapper
pydub>=0.25.1                 # Audio manipulation
scipy>=1.11.0                 # Peak detection (find_peaks)
numpy>=1.24.0                 # Numerics
python-dotenv>=1.0.0          # Env vars
torch>=2.0.0                  # WhisperX backend
```

> **Note**: WhisperX는 pip에서 직접 설치가 불안정하여 requirements.txt에 미포함. 별도 설치 필요: `pip install git+https://github.com/m-bain/whisperX.git`

### System
- FFmpeg: `homebrew-ffmpeg` tap (이미 longform_agent에서 설치됨)
- Python 3.13, Apple M3 Pro

---

## 9. longform_agent 재사용 패턴

| 패턴 | 원본 파일 | 재사용 |
|------|----------|--------|
| Config 구조 | `longform/.../config.py` | path management, retry 설정, VIDEO_PRESETS 딕셔너리 |
| Pipeline resume | `longform/.../pipeline.py` | progress.json save/load, step-by-step 실행 |
| FFmpeg subtitles | `longform/.../video_composer.py` | subtitles/drawtext fallback 패턴 |
| SRT parsing | `longform/.../video_composer.py` | SRT 파싱 유틸리티 |
| Gemini API retry | `longform/.../subtitle_sync_gemini.py` | 지수 백오프 + response null-check |
| .claude/ 구조 | `longform/.claude/` | settings.json, SKILL.md frontmatter 형식 |

---

## 10. Claude Code 통합: 에이전트 & 스킬 설계

longform_agent의 패턴을 따른다: **스킬 = 단일 도구 능력**, **에이전트 = 스킬을 조합하는 오케스트레이터**, **커맨드 = 사용자 인터페이스**.

### 10.1 스킬 정의 (5개)

| 스킬 | 대응 모듈 | 입력 | 출력 |
|------|----------|------|------|
| **download-video** | `downloader.py` | YouTube URL | video.mp4, info.json (heatmap 포함) |
| **transcribe-audio** | `transcriber.py` | audio/video 파일 | words.json, SRT |
| **analyze-segments** | `segment_selector.py` | heatmap + transcript + config | segments.json |
| **crop-vertical** | `video_cropper.py` | source video + segment 정보 | cropped 9:16 MP4 |
| **overlay-subtitle** | `subtitle_overlay.py` + `subtitle_translator.py` | cropped video + SRT | 자막 포함 최종 MP4 |

### 10.2 커맨드

| 커맨드 | 설명 |
|--------|------|
| `/shorts <url>` | YouTube URL → 숏폼 영상 생성 |
| `/analyze <url>` | YouTube URL → 바이럴 구간 분석만 |

---

## 11. 전체 파일 구조

```
260207_shortform_agent/
│
├── .env                              # GEMINI_API_KEY (gitignore)
├── .env.example                      # GEMINI_API_KEY=your_key_here
├── .gitignore
├── requirements.txt                  # Python dependencies
├── PRD.md                            # 이 문서
│
├── src/
│   ├── __init__.py
│   └── shorts_pipeline/
│       ├── __init__.py               # Public API: ShortsPipeline + 각 모듈 클래스
│       ├── config.py                 # 중앙 설정 (경로, API, 폰트, 자막, 번역)
│       ├── downloader.py             # VideoDownloader
│       ├── transcriber.py            # AudioTranscriber (WhisperX, cpu/int8)
│       ├── segment_selector.py       # SegmentSelector (heatmap + Gemini 2-signal)
│       ├── video_cropper.py          # VideoCropper (center crop 16:9→9:16)
│       ├── subtitle_overlay.py       # SubtitleOverlay (3종 스타일, 이중 자막, fallback)
│       ├── subtitle_translator.py    # SubtitleTranslator (Gemini 배치 번역)
│       └── pipeline.py               # ShortsPipeline (오케스트레이션 + resume)
│
├── assets/
│   └── font/                         # 9개 번들 + 1개 시스템 = 10종 프리셋
│       ├── BlackHanSans-Regular.ttf
│       ├── DoHyeon-Regular.ttf
│       ├── Gugi-Regular.ttf
│       ├── IBMPlexSansKR-Bold.ttf
│       ├── Jua-Regular.ttf
│       ├── KOTRA_SONGEULSSI.otf
│       ├── NanumGothic-Bold.ttf
│       ├── NanumPenScript-Regular.ttf
│       └── NotoSansKR-Bold.ttf
│
├── temp/                             # 중간 파일 (gitignore)
│   └── run_YYYYMMDD_HHMMSS/
│       ├── progress.json
│       ├── source/
│       │   ├── video.mp4
│       │   └── info.json
│       ├── transcript/
│       │   ├── words.json
│       │   └── full.srt
│       ├── analysis/
│       │   └── segments.json
│       ├── cropped/
│       │   └── short_001.mp4 ...
│       └── subtitles/
│           └── short_001.srt ...
│
├── output/                           # 최종 결과물
│   ├── VIDEO_ID_short_001.mp4
│   └── VIDEO_ID_metadata.json
│
└── .claude/                          # Claude Code 통합
    ├── settings.json
    └── skills/
        ├── download-video/SKILL.md
        ├── transcribe-audio/SKILL.md
        ├── analyze-segments/SKILL.md
        ├── crop-vertical/SKILL.md
        └── overlay-subtitle/SKILL.md
```

---

## 12. 주의사항

1. **WhisperX**: CTranslate2는 MPS 미지원 → `device="cpu"`, `compute_type="int8"` 필수
2. **PyTorch 2.6+**: `torch.load` monkey-patch 필요 (`kwargs["weights_only"]=False` 강제)
3. **Gemini 2.5 Flash**: thinking model → `max_output_tokens` 8192+, `response_mime_type="application/json"`
4. **yt-dlp heatmap**: `info.json`의 `heatmap` 필드. "Most Replayed" 데이터 없으면 None → fallback weights 사용
5. **FFmpeg center crop 수식**: `crop=ih*9/16:ih:(iw-ih*9/16)/2:0,scale=1080:1920`
6. **FFmpeg path escape**: `replace("\\", "/").replace(":", "\\:")` — macOS 경로에 `:` 포함 가능
7. **google-genai** SDK 사용 (google-generativeai 아님) — longform_agent와 동일
8. **Gemini response**: 항상 `response.candidates[0].content`의 None 체크 후 `.parts` 접근
9. **번역 fallback**: Gemini 번역 API 실패 시 원문만 표시 (이중 자막 → 단일 자막으로 graceful degradation)
10. **자막 fontsize**: 58px (모바일 가독성 강화). 이중 자막 시 원문 43px / 번역 58px
