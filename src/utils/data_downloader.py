"""
Data Downloader for AI Educational Videos (Bilibili - Search Based)
=====================================================================
Uses yt-dlp's bilisearch feature to find and download videos by keyword.
This avoids hardcoded BV numbers that may expire.

Bilibili search syntax: bilisearch<N>:<keyword>  (N = max results)

Video categories:
  - Group A (Golden): High-quality animated/CGI educational videos
  - Group B (Regular): Standard lecture / screencast style videos

Usage (CLI):
    python src/utils/data_downloader.py --group golden --max 5
    python src/utils/data_downloader.py --group regular --max 5

Usage (Colab):
    from src.utils.data_downloader import download_dataset
    download_dataset(output_dir="data/raw_videos", group="all", max_videos=5)
"""

import os
import json
import argparse
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Search-based video definitions
# Using bilisearch: avoids depending on specific BV numbers that may expire.
# Each entry defines a search query; yt-dlp picks the top result.
# ---------------------------------------------------------------------------

GOLDEN_SEARCHES = [
    # 3Blue1Brown 系列 - 精良 Manim 数学动画
    {
        "query": "bilisearch1:3Blue1Brown 神经网络",
        "title": "neural_network_3b1b_zh",
        "description": "3Blue1Brown 神经网络动画教学",
    },
    {
        "query": "bilisearch1:3Blue1Brown 梯度下降",
        "title": "gradient_descent_3b1b_zh",
        "description": "3Blue1Brown 梯度下降动画",
    },
    {
        "query": "bilisearch1:3Blue1Brown 线性代数本质",
        "title": "linear_algebra_3b1b_zh",
        "description": "3Blue1Brown 线性代数本质系列",
    },
    {
        "query": "bilisearch1:3Blue1Brown 傅里叶变换",
        "title": "fourier_transform_3b1b_zh",
        "description": "3Blue1Brown 傅里叶变换可视化",
    },
    {
        "query": "bilisearch1:3Blue1Brown 反向传播",
        "title": "backpropagation_3b1b_zh",
        "description": "3Blue1Brown 反向传播算法",
    },
    # Kurzgesagt 系列 - Motion Graphics 科普动画
    {
        "query": "bilisearch1:kurzgesagt 免疫系统 动画",
        "title": "kurzgesagt_immune_zh",
        "description": "Kurzgesagt 免疫系统科普动画",
    },
    {
        "query": "bilisearch1:kurzgesagt DNA 动画",
        "title": "kurzgesagt_dna_zh",
        "description": "Kurzgesagt DNA动画",
    },
    # 国内高质量动画教学
    {
        "query": "bilisearch1:transformer注意力机制 动画讲解",
        "title": "transformer_animation_zh",
        "description": "Transformer注意力机制精美动画",
    },
    {
        "query": "bilisearch1:diffusion模型 原理动画",
        "title": "diffusion_model_zh",
        "description": "Diffusion模型原理动画讲解",
    },
    {
        "query": "bilisearch1:微积分 动画 可视化 数学",
        "title": "calculus_visual_zh",
        "description": "微积分可视化动画教学",
    },
]

REGULAR_SEARCHES = [
    # 普通 PPT/白板/录屏风格教学
    {
        "query": "bilisearch1:Python基础教程 录屏讲解",
        "title": "python_lecture_zh",
        "description": "Python基础录屏教学",
    },
    {
        "query": "bilisearch1:高中数学 课堂讲解 白板",
        "title": "math_classroom_zh",
        "description": "高中数学课堂讲解",
    },
    {
        "query": "bilisearch1:高中物理 PPT讲解 课堂",
        "title": "physics_lecture_zh",
        "description": "高中物理PPT讲解",
    },
    {
        "query": "bilisearch1:高中化学 实验讲解 课堂",
        "title": "chemistry_lecture_zh",
        "description": "高中化学课堂教学",
    },
    {
        "query": "bilisearch1:高中生物 知识点讲解 PPT",
        "title": "biology_lecture_zh",
        "description": "高中生物PPT讲解",
    },
    {
        "query": "bilisearch1:线性代数 大学 黑板板书",
        "title": "linear_algebra_lecture_zh",
        "description": "线性代数大学板书讲解",
    },
    {
        "query": "bilisearch1:统计学 基础 PPT 讲解",
        "title": "statistics_lecture_zh",
        "description": "统计学PPT教学",
    },
    {
        "query": "bilisearch1:机器学习 入门 课堂讲解",
        "title": "ml_lecture_zh",
        "description": "机器学习课堂讲解",
    },
    {
        "query": "bilisearch1:英语课堂 教学录像 讲解",
        "title": "english_lecture_zh",
        "description": "英语课堂教学录像",
    },
    {
        "query": "bilisearch1:历史课 PPT讲解 知识点",
        "title": "history_lecture_zh",
        "description": "历史PPT讲解",
    },
]


def _check_ytdlp():
    """Ensure yt-dlp is installed."""
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[INFO] Installing yt-dlp...")
        subprocess.run([sys.executable, "-m", "pip", "install", "yt-dlp", "-q"], check=True)


def download_by_search(
    query: str,
    title: str,
    output_dir: str,
    max_duration_sec: int = 1800,
) -> str | None:
    """
    Download a video using yt-dlp Bilibili search query.

    Args:
        query: bilisearch query, e.g. "bilisearch1:3Blue1Brown 神经网络"
        title: Output filename stem (no extension)
        output_dir: Directory to save video
        max_duration_sec: Skip videos longer than this

    Returns:
        Path to downloaded file or None.
    """
    output_path = os.path.join(output_dir, f"{title}.mp4")

    if os.path.exists(output_path):
        print(f"  [SKIP] Already exists: {output_path}")
        return output_path

    print(f"  Searching & downloading: {title} ...")
    print(f"    Query: {query}")

    cmd = [
        "yt-dlp",
        "--format", "bestvideo[height<=720][ext=mp4]+bestaudio/best[height<=720]/best",
        "--merge-output-format", "mp4",
        "--output", output_path,
        "--match-filter", f"duration < {max_duration_sec}",
        "--no-playlist",
        "--quiet",
        "--progress",
        # --- Anti-412 headers: Bilibili requires Referer & real UA ---
        "--add-header", "Referer:https://www.bilibili.com",
        "--add-header", "Origin:https://www.bilibili.com",
        "--user-agent", (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        # Random sleep 2-5s between requests to avoid rate-limit
        "--sleep-interval", "2",
        "--max-sleep-interval", "5",
        query,
    ]

    try:
        subprocess.run(cmd, check=True, timeout=600)
        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  [OK] {title}.mp4 ({size_mb:.1f} MB)")
            return output_path
        else:
            print(f"  [WARN] File not created for: {title}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"  [FAIL] {title}: {e}")
        return None
    except subprocess.TimeoutExpired:
        print(f"  [FAIL] Timeout: {title}")
        return None


def download_dataset(
    output_dir: str = "data/raw_videos",
    group: str = "all",
    max_videos: int = 5,
) -> dict:
    """
    Download educational video dataset from Bilibili using keyword search.

    Args:
        output_dir: Root directory (subfolders: golden/, regular/)
        group: "golden", "regular", or "all"
        max_videos: Max videos per group

    Returns:
        dict with "golden" and "regular" lists of downloaded paths.
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
        for item in GOLDEN_SEARCHES[:max_videos]:
            path = download_by_search(item["query"], item["title"], golden_dir)
            if path:
                results["golden"].append(path)
                manifest[os.path.basename(path)] = {
                    "group": "golden",
                    "description": item["description"],
                    "query": item["query"],
                }

    if group in ("regular", "all"):
        print(f"\n=== Downloading Regular Group (up to {max_videos} videos) ===")
        for item in REGULAR_SEARCHES[:max_videos]:
            path = download_by_search(item["query"], item["title"], regular_dir)
            if path:
                results["regular"].append(path)
                manifest[os.path.basename(path)] = {
                    "group": "regular",
                    "description": item["description"],
                    "query": item["query"],
                }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n=== Download Complete ===")
    print(f"  Golden : {len(results['golden'])}")
    print(f"  Regular: {len(results['regular'])}")
    print(f"  Manifest: {manifest_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Bilibili educational videos by search")
    parser.add_argument("--group", choices=["golden", "regular", "all"], default="all")
    parser.add_argument("--max", type=int, default=5)
    parser.add_argument("--output", default="data/raw_videos")
    args = parser.parse_args()
    download_dataset(output_dir=args.output, group=args.group, max_videos=args.max)
