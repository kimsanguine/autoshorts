"""
Stage 1 테스트: YouTube 다운로드 + heatmap 추출
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

from shorts_pipeline import VideoDownloader


def test_download():
    """짧은 YouTube 영상으로 다운로드 + heatmap 테스트."""

    # 조회수 높은 짧은 영상 (heatmap 있을 확률 높음)
    # 테스트용: TED-Ed 짧은 영상
    test_url = "https://www.youtube.com/watch?v=arj7oStGLkU"

    print("=" * 60)
    print("Stage 1 Test: Download + Heatmap Extraction")
    print("=" * 60)

    downloader = VideoDownloader()
    result = downloader.download(test_url)

    print(f"\n{'=' * 60}")
    print(f"Results:")
    print(f"  Title:      {result['title']}")
    print(f"  Video ID:   {result['video_id']}")
    print(f"  Duration:   {result['duration']:.1f}s")
    print(f"  Views:      {result['view_count']:,}")
    print(f"  Video:      {result['video_path']}")
    print(f"  Info JSON:  {result['info_path']}")

    if result['heatmap']:
        print(f"  Heatmap:    {len(result['heatmap'])} data points")
        # Show top 3 peaks
        sorted_hm = sorted(result['heatmap'], key=lambda x: x.get('value', 0), reverse=True)
        print(f"  Top peaks:")
        for i, entry in enumerate(sorted_hm[:3]):
            print(f"    #{i+1}: {entry.get('start_time', 0):.1f}s - {entry.get('end_time', 0):.1f}s (value: {entry.get('value', 0):.3f})")
    else:
        print(f"  Heatmap:    Not available")

    if result['chapters']:
        print(f"  Chapters:   {len(result['chapters'])}")
    else:
        print(f"  Chapters:   None")

    print(f"\n{'=' * 60}")
    print("Stage 1 Test PASSED")

    return result


if __name__ == "__main__":
    test_download()
