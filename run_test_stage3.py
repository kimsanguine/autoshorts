"""
Stage 3 테스트: Segment Selection (Heatmap + Gemini Content Analysis)
기존 Stage 1-2 결과(다운로드 + 트랜스크립트)를 재활용하여 세그먼트 선택을 테스트한다.
"""

import sys
import json
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

from shorts_pipeline import SegmentSelector


def test_segment_selection():
    """기존 다운로드/트랜스크립트 결과로 세그먼트 선택 테스트."""

    # Stage 1-2에서 생성된 데이터 경로
    run_dir = Path(__file__).parent / "temp" / "run_20260207_102153"
    info_path = run_dir / "source" / "arj7oStGLkU.info.json"
    words_path = run_dir / "transcript" / "arj7oStGLkU_words.json"

    if not info_path.exists() or not words_path.exists():
        print("ERROR: Stage 1-2 test data not found. Run run_test.py first.")
        sys.exit(1)

    print("=" * 60)
    print("Stage 3 Test: Segment Selection")
    print("=" * 60)

    # Load heatmap
    with open(info_path, "r") as f:
        info = json.load(f)
    heatmap = info.get("heatmap")
    duration = info.get("duration", 0)

    print(f"  Heatmap: {len(heatmap) if heatmap else 0} data points")
    print(f"  Duration: {duration:.1f}s")

    # Load transcript
    with open(words_path, "r") as f:
        transcript = json.load(f)

    print(f"  Words: {transcript['word_count']}")
    print(f"  Segments: {len(transcript['segments'])}")

    # Run segment selection
    print(f"\n{'=' * 60}")
    print("Running segment analysis...")
    print("=" * 60)

    selector = SegmentSelector(n_shorts=5)
    analysis_dir = run_dir / "analysis"

    segments = selector.analyze(
        heatmap=heatmap,
        transcript=transcript,
        video_duration=duration,
        output_dir=analysis_dir,
    )

    # Display results
    print(f"\n{'=' * 60}")
    print(f"Results: {len(segments)} segments selected")
    print("=" * 60)

    for i, seg in enumerate(segments):
        start_m, start_s = divmod(int(seg["start"]), 60)
        end_m, end_s = divmod(int(seg["end"]), 60)
        duration_s = seg["end"] - seg["start"]
        breakdown = seg.get("score_breakdown", {})

        print(f"\n  Segment #{i+1}:")
        print(f"    Time:    {start_m:02d}:{start_s:02d} - {end_m:02d}:{end_s:02d} ({duration_s:.1f}s)")
        print(f"    Score:   {seg['score']:.3f}")
        print(f"    Heatmap: {breakdown.get('heatmap', 0):.3f}")
        print(f"    Content: {breakdown.get('content', 0):.3f}")
        print(f"    Reason:  {seg['reason']}")

        # Show words in this segment
        seg_words = [
            w for w in transcript["words"]
            if w.get("start", 0) >= seg["start"] and w.get("end", 0) <= seg["end"]
        ]
        if seg_words:
            text_preview = " ".join(w["word"] for w in seg_words[:20])
            if len(seg_words) > 20:
                text_preview += "..."
            print(f"    Text:    {text_preview}")

    print(f"\n{'=' * 60}")
    print(f"Segments JSON saved to: {analysis_dir / 'segments.json'}")
    print("Stage 3 Test PASSED")

    return segments


if __name__ == "__main__":
    test_segment_selection()
