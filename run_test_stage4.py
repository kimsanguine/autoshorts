"""
Stage 4 테스트: Video Crop + Subtitle Overlay
기존 Stage 1-3 결과를 재활용하여 크롭 + 자막 합성 테스트.
1개 세그먼트만 테스트한다 (속도).
"""

import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

from shorts_pipeline import VideoCropper, SubtitleOverlay, AudioTranscriber


def test_crop_and_subtitle():
    """크롭 + 자막 오버레이 테스트 (1개 세그먼트)."""

    run_dir = Path(__file__).parent / "temp" / "run_20260207_102153"
    video_path = run_dir / "source" / "arj7oStGLkU.mp4"
    words_path = run_dir / "transcript" / "arj7oStGLkU_words.json"
    segments_path = run_dir / "analysis" / "segments.json"

    for p in [video_path, words_path, segments_path]:
        if not p.exists():
            print(f"ERROR: {p} not found. Run previous stages first.")
            sys.exit(1)

    print("=" * 60)
    print("Stage 4 Test: Video Crop + Subtitle Overlay")
    print("=" * 60)

    # Load data
    with open(words_path) as f:
        transcript = json.load(f)
    with open(segments_path) as f:
        analysis = json.load(f)

    segments = analysis["segments"]
    words = transcript["words"]

    # Test with first segment only
    seg = segments[0]
    start_m, start_s = divmod(int(seg["start"]), 60)
    end_m, end_s = divmod(int(seg["end"]), 60)
    print(f"\n  Test segment: {start_m:02d}:{start_s:02d} - {end_m:02d}:{end_s:02d}")
    print(f"  Score: {seg['score']:.3f}")
    print(f"  Reason: {seg['reason']}")

    # Step 1: Crop
    print(f"\n{'=' * 60}")
    print("Step 1: Center Crop (16:9 → 9:16)")
    print("=" * 60)

    cropper = VideoCropper()
    cropped_dir = run_dir / "cropped"
    cropped_path = cropped_dir / "arj7oStGLkU_short_001.mp4"

    cropper.crop_segment(
        video_path=video_path,
        start=seg["start"],
        end=seg["end"],
        output_path=cropped_path,
    )

    # Verify crop result
    import subprocess
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,duration",
         "-of", "csv=p=0", str(cropped_path)],
        capture_output=True, text=True,
    )
    print(f"  Cropped video: {cropped_path.name}")
    print(f"  Resolution/Duration: {probe.stdout.strip()}")

    # Step 2: Generate segment SRT
    print(f"\n{'=' * 60}")
    print("Step 2: Generate Segment SRT")
    print("=" * 60)

    transcriber = AudioTranscriber()
    srt_dir = run_dir / "subtitles"
    srt_path = srt_dir / "arj7oStGLkU_short_001.srt"

    transcriber.generate_segment_srt(
        words=words,
        start_time=seg["start"],
        end_time=seg["end"],
        output_path=srt_path,
    )

    with open(srt_path) as f:
        srt_content = f.read()
    print(f"  SRT file: {srt_path.name}")
    print(f"  SRT preview:\n{srt_content[:300]}")

    # Step 3: Subtitle overlay
    print(f"\n{'=' * 60}")
    print("Step 3: Subtitle Overlay")
    print("=" * 60)

    overlay = SubtitleOverlay()
    output_dir = run_dir / "output"
    output_path = output_dir / "arj7oStGLkU_short_001.mp4"

    overlay.apply(
        video_path=cropped_path,
        srt_path=srt_path,
        output_path=output_path,
    )

    # Verify final output
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"  Final output: {output_path}")
    print(f"  File size: {file_size:.1f} MB")

    probe2 = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,duration",
         "-of", "csv=p=0", str(output_path)],
        capture_output=True, text=True,
    )
    print(f"  Resolution/Duration: {probe2.stdout.strip()}")

    print(f"\n{'=' * 60}")
    print("Stage 4 Test PASSED")
    print(f"Output: {output_path}")

    return output_path


if __name__ == "__main__":
    test_crop_and_subtitle()
