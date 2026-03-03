"""
Data Downloader for AI-Generated Educational Videos
=====================================================
Uses yt-dlp to download AI-generated / AIGC educational videos from YouTube.

Video categories collected:
  - Group A (Golden): Award-winning, highly-viewed AIGC educational explainers
  - Group B (Regular): Standard educational videos

Usage (CLI):
    python src/utils/data_downloader.py --group golden --max 10
    python src/utils/data_downloader.py --group regular --max 10
    python src/utils/data_downloader.py --all --max 20

Usage (Colab):
    from src.utils.data_downloader import download_dataset
    download_dataset(output_dir="data/raw_videos", group="golden", max_videos=10)
"""

import os
import json
import argparse
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Curated video lists
# These are publicly available educational videos on YouTube that exhibit
# characteristics of AI-generated or AI-assisted production.
# ---------------------------------------------------------------------------

GOLDEN_VIDEOS = [
    # High-quality AIGC / AI-assisted educational explainers
    # (cinematic lighting, structured animation, professional narration)
    {
        "url": "https://www.youtube.com/watch?v=aircAruvnKk",
        "title": "neural_network_3b1b",
        "description": "3Blue1Brown: But what is a Neural Network? (Premium animation quality)",
    },
    {
        "url": "https://www.youtube.com/watch?v=IHZwWFHWa-w",
        "title": "gradient_descent_3b1b",
        "description": "3Blue1Brown: Gradient descent (Cinematic math animation)",
    },
    {
        "url": "https://www.youtube.com/watch?v=9-Jl0dxWQs8",
        "title": "backpropagation_3b1b",
        "description": "3Blue1Brown: Backpropagation (High-quality explainer)",
    },
    {
        "url": "https://www.youtube.com/watch?v=qv6UVOQ0F44",
        "title": "dna_animation_premium",
        "description": "DNA replication animation (Professional CGI)",
    },
    {
        "url": "https://www.youtube.com/watch?v=SmZmBKc7Lrs",
        "title": "cell_biology_animation",
        "description": "Inner Life of a Cell (Award-winning CGI animation)",
    },
    {
        "url": "https://www.youtube.com/watch?v=PtKnpN3xg4I",
        "title": "kurzgesagt_immune",
        "description": "Kurzgesagt: The Immune System (Premium motion graphics)",
    },
    {
        "url": "https://www.youtube.com/watch?v=GNcFjFmqEc8",
        "title": "kurzgesagt_mitosis",
        "description": "Kurzgesagt educational style animation",
    },
    {
        "url": "https://www.youtube.com/watch?v=sBykYQ7gExg",
        "title": "photosynthesis_animation",
        "description": "Photosynthesis process animation (CGI quality)",
    },
    {
        "url": "https://www.youtube.com/watch?v=K8LQhy8uGrM",
        "title": "fourier_transform_visual",
        "description": "Fourier Transform visual explainer",
    },
    {
        "url": "https://www.youtube.com/watch?v=spUNpyF58BY",
        "title": "euler_formula_3b1b",
        "description": "3Blue1Brown: Euler's formula (Manim animation)",
    },
]

REGULAR_VIDEOS = [
    # Standard classroom / screencast style educational videos
    {
        "url": "https://www.youtube.com/watch?v=MUDoR0MLFsQ",
        "title": "basic_python_tutorial",
        "description": "Basic Python tutorial (screen recording style)",
    },
    {
        "url": "https://www.youtube.com/watch?v=rfscVS0vtbw",
        "title": "python_course_fcc",
        "description": "FreeCodeCamp Python course (talking head + screen)",
    },
    {
        "url": "https://www.youtube.com/watch?v=kqtD5dpn9C8",
        "title": "python_for_beginners",
        "description": "Standard Python beginners course",
    },
    {
        "url": "https://www.youtube.com/watch?v=Tia4am0caOS",
        "title": "photosynthesis_classroom",
        "description": "Classroom-style photosynthesis explanation",
    },
    {
        "url": "https://www.youtube.com/watch?v=zNCsiV9p0JU",
        "title": "math_whiteboard",
        "description": "Whiteboard math lecture style",
    },
    {
        "url": "https://www.youtube.com/watch?v=GFD1NadZ0AA",
        "title": "physics_lecture_basic",
        "description": "Basic physics lecture (slides + voiceover)",
    },
    {
        "url": "https://www.youtube.com/watch?v=l9AzO1FMgM8",
        "title": "chemistry_basics_lecture",
        "description": "Basic chemistry explanations",
    },
    {
        "url": "https://www.youtube.com/watch?v=Vwj0ZQNVJNQ",
        "title": "biology_basics",
        "description": "Biology basics (lecture style)",
    },
    {
        "url": "https://www.youtube.com/watch?v=8mAITcNt710",
        "title": "algebra_basics",
        "description": "Algebra basics (whiteboard style)",
    },
    {
        "url": "https://www.youtube.com/watch?v=X18zUDMCrBo",
        "title": "statistics_intro",
        "description": "Statistics introduction (talking head style)",
    },
]


def _check_ytdlp():
    """Check if yt-dlp is installed."""
    try:
        subprocess.run(
            ["yt-dlp", "--version"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[INFO] yt-dlp not found. Installing...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "yt-dlp", "-q"],
            check=True,
        )
        return True


def download_video(url: str, title: str, output_dir: str, max_duration_sec: int = 600) -> str | None:
    """
    Download a single video using yt-dlp.
    
    Args:
        url: YouTube video URL
        title: Filename (without extension)
        output_dir: Directory to save the video
        max_duration_sec: Skip videos longer than this (default 10 min)
    
    Returns:
        Path to downloaded file, or None if failed.
    """
    output_path = os.path.join(output_dir, f"{title}.mp4")
    
    if os.path.exists(output_path):
        print(f"  [SKIP] Already exists: {output_path}")
        return output_path
    
    cmd = [
        "yt-dlp",
        "--format", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]",
        "--merge-output-format", "mp4",
        "--output", output_path,
        "--match-filter", f"duration < {max_duration_sec}",
        "--no-playlist",
        "--quiet",
        "--progress",
        url,
    ]
    
    print(f"  Downloading: {title} ...")
    try:
        subprocess.run(cmd, check=True, timeout=300)
        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  [OK] {output_path} ({size_mb:.1f} MB)")
            return output_path
        else:
            print(f"  [WARN] File not created: {output_path}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"  [FAIL] Download error for {title}: {e}")
        return None
    except subprocess.TimeoutExpired:
        print(f"  [FAIL] Timeout for {title}")
        return None


def download_dataset(
    output_dir: str = "data/raw_videos",
    group: str = "all",
    max_videos: int = 10,
) -> dict:
    """
    Download the curated dataset.
    
    Args:
        output_dir: Root directory for videos (subfolders: golden/, regular/)
        group: "golden", "regular", or "all"
        max_videos: Max videos per group to download
    
    Returns:
        dict with keys "golden" and "regular", each a list of downloaded paths.
    """
    _check_ytdlp()
    
    golden_dir = os.path.join(output_dir, "golden")
    regular_dir = os.path.join(output_dir, "regular")
    os.makedirs(golden_dir, exist_ok=True)
    os.makedirs(regular_dir, exist_ok=True)
    
    results = {"golden": [], "regular": []}
    manifest = {}
    
    if group in ("golden", "all"):
        print(f"\n=== Downloading Golden Group (up to {max_videos} videos) ===")
        for i, video in enumerate(GOLDEN_VIDEOS[:max_videos]):
            path = download_video(video["url"], video["title"], golden_dir)
            if path:
                results["golden"].append(path)
                manifest[os.path.basename(path)] = {
                    "group": "golden",
                    "description": video["description"],
                    "url": video["url"],
                }
    
    if group in ("regular", "all"):
        print(f"\n=== Downloading Regular Group (up to {max_videos} videos) ===")
        for i, video in enumerate(REGULAR_VIDEOS[:max_videos]):
            path = download_video(video["url"], video["title"], regular_dir)
            if path:
                results["regular"].append(path)
                manifest[os.path.basename(path)] = {
                    "group": "regular",
                    "description": video["description"],
                    "url": video["url"],
                }
    
    # Save manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Download Complete ===")
    print(f"  Golden videos: {len(results['golden'])}")
    print(f"  Regular videos: {len(results['regular'])}")
    print(f"  Manifest saved to: {manifest_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download educational video dataset")
    parser.add_argument(
        "--group",
        choices=["golden", "regular", "all"],
        default="all",
        help="Which group to download",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=5,
        help="Max videos per group (default: 5 for quick test)",
    )
    parser.add_argument(
        "--output",
        default="data/raw_videos",
        help="Output directory (default: data/raw_videos)",
    )
    
    args = parser.parse_args()
    download_dataset(output_dir=args.output, group=args.group, max_videos=args.max)
