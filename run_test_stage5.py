"""
Stage 5 테스트: Full Pipeline (resume 기능으로 기존 데이터 활용)
기존 run_20260207_102153의 Step 1-3 결과를 활용하여 전체 파이프라인을 resume 테스트.
"""

import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

from shorts_pipeline import ShortsPipeline


def test_pipeline_resume():
    """기존 run에서 resume하여 crop → subtitle → finalize 실행."""

    run_dir = Path(__file__).parent / "temp" / "run_20260207_102153"
    progress_path = run_dir / "progress.json"

    # 기존 progress.json이 없으면 생성
    if not progress_path.exists():
        # Stage 1-3 결과를 progress.json으로 기록
        video_id = "arj7oStGLkU"
        info_path = run_dir / "source" / f"{video_id}.info.json"
        words_path = run_dir / "transcript" / f"{video_id}_words.json"
        srt_path = run_dir / "transcript" / f"{video_id}.srt"
        segments_path = run_dir / "analysis" / "segments.json"

        with open(info_path) as f:
            info = json.load(f)
        with open(words_path) as f:
            transcript = json.load(f)

        progress = {
            "url": "https://www.youtube.com/watch?v=arj7oStGLkU",
            "steps": {
                "download": {
                    "status": "done",
                    "video_path": str(run_dir / "source" / f"{video_id}.mp4"),
                    "video_id": video_id,
                    "title": info.get("title", "Inside the mind of a master procrastinator"),
                    "duration": info.get("duration", 844.0),
                    "view_count": info.get("view_count", 0),
                    "has_heatmap": info.get("heatmap") is not None,
                },
                "transcribe": {
                    "status": "done",
                    "words_path": str(words_path),
                    "srt_path": str(srt_path),
                    "word_count": transcript["word_count"],
                    "language": transcript["language"],
                },
                "analyze": {
                    "status": "done",
                    "segments_path": str(segments_path),
                    "n_segments": 5,
                },
            },
            "last_step": "analyze",
        }

        with open(progress_path, "w") as f:
            json.dump(progress, f, indent=2)
        print("Created progress.json from existing Stage 1-3 data")

    print("=" * 60)
    print("Stage 5 Test: Full Pipeline (Resume from Step 4)")
    print("=" * 60)

    pipeline = ShortsPipeline(n_shorts=3)  # 3개만 테스트 (속도)
    output_dir = Path(__file__).parent / "output"

    result = pipeline.generate(
        youtube_url="https://www.youtube.com/watch?v=arj7oStGLkU",
        output_dir=output_dir,
        keep_temp_files=True,
        resume_from=run_dir,
    )

    print(f"\n{'=' * 60}")
    print(f"Results: {len(result['output_paths'])} shorts generated")
    print("=" * 60)

    for path in result["output_paths"]:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {path.name} ({size_mb:.1f} MB)")

    print(f"\n  Metadata: {result['metadata_path']}")

    # Print metadata summary
    meta = result["metadata"]
    print(f"\n  Source: {meta['source']['title']}")
    print(f"  Duration: {meta['source']['duration']:.0f}s")
    print(f"  Heatmap: {'Yes' if meta['source']['has_heatmap'] else 'No'}")

    for short in meta["shorts"]:
        start_m, start_s = divmod(int(short["source_start"]), 60)
        end_m, end_s = divmod(int(short["source_end"]), 60)
        print(
            f"\n  Short #{short['index']+1}: {start_m:02d}:{start_s:02d}-{end_m:02d}:{end_s:02d} "
            f"(score={short['score']:.3f})"
        )
        print(f"    {short['reason']}")

    print(f"\n{'=' * 60}")
    print("Stage 5 Test PASSED")

    return result


if __name__ == "__main__":
    test_pipeline_resume()
