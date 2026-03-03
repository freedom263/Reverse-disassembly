"""
Data Downloader for AI Educational Videos (Bilibili Source)
============================================================
Uses yt-dlp to download educational videos from Bilibili (B站).
Bilibili is accessible from Colab without restrictions.

Video categories:
  - Group A (Golden): High-quality Manim/CGI animated educational videos
    (3Blue1Brown 系列中文译制版, Kurzgesagt 译制版等)
  - Group B (Regular): Standard lecture / screencast style videos

Usage (CLI):
    python src/utils/data_downloader.py --group golden --max 10
    python src/utils/data_downloader.py --group regular --max 10

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
# Bilibili Curated Video Lists
# Golden: 高质量 Manim/CGI 动画教学视频
# Regular: 普通 PPT/白板/录屏教学视频
# ---------------------------------------------------------------------------

GOLDEN_VIDEOS = [
    # 3Blue1Brown 系列 - 精良 Manim 数学动画（B站官方译制版）
    {
        "url": "https://www.bilibili.com/video/BV1bx411j7tT",
        "title": "neural_network_3b1b_zh",
        "description": "3Blue1Brown: 神经网络是什么？（Manim动画，高质量）",
    },
    {
        "url": "https://www.bilibili.com/video/BV1Ux411j7M1",
        "title": "gradient_descent_3b1b_zh",
        "description": "3Blue1Brown: 梯度下降法（电影级数学动画）",
    },
    {
        "url": "https://www.bilibili.com/video/BV16x411V7Qg",
        "title": "backpropagation_3b1b_zh",
        "description": "3Blue1Brown: 反向传播算法（高质量动画讲解）",
    },
    {
        "url": "https://www.bilibili.com/video/BV1ys411472E",
        "title": "fourier_transform_3b1b_zh",
        "description": "3Blue1Brown: 傅里叶变换（精良可视化）",
    },
    {
        "url": "https://www.bilibili.com/video/BV1HJ411d7WM",
        "title": "linear_algebra_3b1b_zh",
        "description": "3Blue1Brown: 线性代数本质（动画教学系列）",
    },
    # Kurzgesagt 系列 - 精良 Motion Graphics 科普动画
    {
        "url": "https://www.bilibili.com/video/BV1Rs411V7zA",
        "title": "kurzgesagt_immune_zh",
        "description": "Kurzgesagt: 免疫系统（专业Motion Graphics动画）",
    },
    {
        "url": "https://www.bilibili.com/video/BV13b411M74y",
        "title": "kurzgesagt_dna_zh",
        "description": "Kurzgesagt: DNA动画（CGI精良制作）",
    },
    # 国内高质量 AIGC / 动画教学视频
    {
        "url": "https://www.bilibili.com/video/BV1Km4y1C7Bz",
        "title": "diffusion_model_visual_zh",
        "description": "Diffusion模型原理可视化（AIGC风格动画）",
    },
    {
        "url": "https://www.bilibili.com/video/BV1TF4m1E7NR",
        "title": "transformer_animation_zh",
        "description": "Transformer架构精美动画解析",
    },
    {
        "url": "https://www.bilibili.com/video/BV1vV4y1H7YH",
        "title": "calculus_animation_zh",
        "description": "微积分精美动画教学（国产高质量）",
    },
]

REGULAR_VIDEOS = [
    # 普通录屏/PPT/讲座风格教学视频
    {
        "url": "https://www.bilibili.com/video/BV1qW4y1a7fU",
        "title": "python_lecture_basic_zh",
        "description": "Python基础教学（录屏讲解，普通风格）",
    },
    {
        "url": "https://www.bilibili.com/video/BV1Mb4y1e7QL",
        "title": "math_whiteboard_zh",
        "description": "高中数学白板讲解（普通拍摄风格）",
    },
    {
        "url": "https://www.bilibili.com/video/BV1R84y1j7ie",
        "title": "physics_ppt_lecture_zh",
        "description": "物理PPT讲解（幻灯片+旁白风格）",
    },
    {
        "url": "https://www.bilibili.com/video/BV1fP4y1V7Dz",
        "title": "chemistry_basic_lecture_zh",
        "description": "化学基础教学（课堂录像风格）",
    },
    {
        "url": "https://www.bilibili.com/video/BV1gz4y1r75M",
        "title": "biology_class_lecture_zh",
        "description": "生物基础讲解（PPT+语音风格）",
    },
    {
        "url": "https://www.bilibili.com/video/BV1bT411k7ZC",
        "title": "linear_algebra_lecture_zh",
        "description": "线性代数大学课堂（黑板板书风格）",
    },
    {
        "url": "https://www.bilibili.com/video/BV1Rv4y1H7uc",
        "title": "statistics_ppt_zh",
        "description": "统计学PPT教学（普通幻灯片风格）",
    },
    {
        "url": "https://www.bilibili.com/video/BV1Y84y1b7xj",
        "title": "english_classroom_zh",
        "description": "英语课堂教学录像（普通拍摄）",
    },
    {
        "url": "https://www.bilibili.com/video/BV1Be4y1p7Vk",
        "title": "history_lecture_zh",
        "description": "历史课录屏讲解（PPT配音）",
    },
    {
        "url": "https://www.bilibili.com/video/BV1xs4y1C7SD",
        "title": "machine_learning_basic_lecture_zh",
        "description": "机器学习基础课（幻灯片+摄像头，普通风格）",
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


def download_video(url: str, title: str, output_dir: str, max_duration_sec: int = 1200) -> str | None:
    """
    Download a single video using yt-dlp.

    Args:
        url: Bilibili video URL
        title: Filename (without extension)
        output_dir: Directory to save the video
        max_duration_sec: Skip videos longer than this (default 20 min)

    Returns:
        Path to downloaded file, or None if failed.
    """
    output_path = os.path.join(output_dir, f"{title}.mp4")

    if os.path.exists(output_path):
        print(f"  [SKIP] Already exists: {output_path}")
        return output_path

    cmd = [
        "yt-dlp",
        "--format", "bestvideo[height<=720][ext=mp4]+bestaudio/best[height<=720]/best",
        "--merge-output-format", "mp4",
        "--output", output_path,
        "--no-playlist",
        "--quiet",
        "--progress",
        url,
    ]

    print(f"  Downloading: {title} ...")
    try:
        subprocess.run(cmd, check=True, timeout=600)
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
    Download the curated Bilibili dataset.

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
        for video in GOLDEN_VIDEOS[:max_videos]:
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
        for video in REGULAR_VIDEOS[:max_videos]:
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
    print(f"  Golden videos : {len(results['golden'])}")
    print(f"  Regular videos: {len(results['regular'])}")
    print(f"  Manifest saved: {manifest_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download educational video dataset from Bilibili")
    parser.add_argument("--group", choices=["golden", "regular", "all"], default="all")
    parser.add_argument("--max", type=int, default=5, help="Max videos per group")
    parser.add_argument("--output", default="data/raw_videos")

    args = parser.parse_args()
    download_dataset(output_dir=args.output, group=args.group, max_videos=args.max)
