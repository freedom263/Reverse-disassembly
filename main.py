"""
Main Pipeline - Reverse Prompt Engineering for AI Educational Videos
=====================================================================
CLI entry point for the full reverse-prompt pipeline.

Usage:
    # Single video:
    python main.py --video data/raw_videos/golden/neural_network_3b1b.mp4

    # Batch processing a directory:
    python main.py --batch data/raw_videos/golden/ --group golden

    # Full pipeline (download + process + analyze):
    python main.py --download --max 5 --batch data/raw_videos/ --analyze

    # Pattern mining only (from existing results):
    python main.py --analyze --results results/
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Lazy imports (avoid loading GPU models unless needed)
# ---------------------------------------------------------------------------


def run_single_video(
    video_path: str,
    output_dir: str,
    group: str = "golden",
    lang: str = "en",
    keyframe_dir: str = None,
) -> dict:
    """
    Full pipeline for a single video:
    video → keyframes → VLM analysis → agents → synthesized prompt

    Returns:
        dict: The synthesized prompt result.
    """
    from src.utils.video_processor import process_video
    from src.perception.vlm_analyzer import VLMAnalyzer
    from src.agents.pedagogy_agent import PedagogyAgent
    from src.agents.art_agent import ArtDirectorAgent
    from src.agents.structure_agent import StructureAgent
    from src.agents.prompt_synthesizer import PromptSynthesizer

    video_name = Path(video_path).stem
    if keyframe_dir is None:
        keyframe_dir = os.path.join("data", "keyframes", video_name)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(keyframe_dir, exist_ok=True)

    # ---------- Stage 1: Video → Keyframes ----------
    print(f"\n{'='*60}")
    print(f"[Pipeline] Processing: {video_path}")
    print(f"{'='*60}")

    keyframes = process_video(video_path, keyframe_dir)
    if not keyframes:
        print(f"[Pipeline] ERROR: No keyframes extracted from {video_path}")
        return {}

    print(f"[Pipeline] Extracted {len(keyframes)} keyframes")

    # ---------- Stage 2: VLM Analysis (use first 3 keyframes for efficiency) ----------
    print("\n[Pipeline] Stage 2: VLM Frame Analysis")
    vlm = VLMAnalyzer(lang=lang)

    # Analyze up to 3 representative frames
    sample_frames = keyframes[:3]
    vlm_results = vlm.analyze_batch(sample_frames)

    # Use the first successful result as representative
    vlm_output = next(
        (r for r in vlm_results if "_error" not in r),
        vlm_results[0] if vlm_results else {}
    )
    print(f"[Pipeline] VLM Result: {json.dumps(vlm_output, ensure_ascii=False)[:200]}...")

    # ---------- Stage 3: Multi-Agent Reasoning ----------
    print("\n[Pipeline] Stage 3: Multi-Agent Reasoning")

    pedagogy_agent = PedagogyAgent()
    art_agent = ArtDirectorAgent()
    structure_agent = StructureAgent()

    pedagogy_intent = pedagogy_agent.analyze(vlm_output)
    print(f"  Pedagogy: {pedagogy_intent[:100]}...")

    art_style = art_agent.analyze(vlm_output)
    print(f"  Art Style: {art_style[:100]}...")

    composition = structure_agent.analyze(vlm_output)
    print(f"  Structure: {composition[:100]}...")

    # ---------- Stage 4: Prompt Synthesis ----------
    print("\n[Pipeline] Stage 4: Prompt Synthesis")
    synthesizer = PromptSynthesizer()
    result = synthesizer.synthesize(
        pedagogy_intent=pedagogy_intent,
        art_style=art_style,
        structure=composition,
        vlm_description=vlm_output,
        source_frame=sample_frames[0] if sample_frames else "",
    )

    # Add group and video metadata
    result["metadata"]["group"] = group
    result["metadata"]["source_video"] = video_path
    result["metadata"]["video_name"] = video_name
    result["metadata"]["keyframes_analyzed"] = len(sample_frames)
    result["metadata"]["total_keyframes"] = len(keyframes)

    # ---------- Save result ----------
    output_path = os.path.join(output_dir, f"{video_name}_prompt.json")
    synthesizer.save_result(result, output_path)

    print(f"\n{'='*60}")
    print("[Pipeline] FINAL REVERSED PROMPT:")
    print(f"{'='*60}")
    print(result["full_prompt"])
    print(f"{'='*60}\n")

    return result


def run_batch(
    video_dir: str,
    output_dir: str,
    group: str = "golden",
    lang: str = "en",
) -> list[dict]:
    """
    Run the pipeline on all MP4 files in a directory.
    """
    video_files = sorted(
        f for f in Path(video_dir).glob("*.mp4")
        if not f.name.endswith("_h264.mp4")  # skip transcoded copies
    )
    if not video_files:
        print(f"[Batch] No MP4 files found in {video_dir}")
        return []

    print(f"[Batch] Found {len(video_files)} videos in {video_dir}")
    results = []

    for i, video_path in enumerate(video_files):
        print(f"\n[Batch] Video {i+1}/{len(video_files)}: {video_path.name}")
        try:
            result = run_single_video(
                str(video_path), output_dir, group=group, lang=lang
            )
            if result:
                results.append(result)
        except Exception as e:
            print(f"[Batch] ERROR processing {video_path.name}: {e}")

    return results


def run_pattern_analysis(results_dir: str):
    """
    Run pattern mining on all saved prompt JSON files.
    """
    from src.analysis.pattern_miner import PatternMiner
    from src.analysis.prompt_ranker import PromptRanker

    print("\n[Analysis] Loading prompts from results directory...")
    miner = PatternMiner()
    golden_prompts, regular_prompts = miner.load_prompts_from_results(results_dir)

    if not golden_prompts and not regular_prompts:
        print("[Analysis] No prompts found. Run the pipeline first.")
        return

    report = miner.analyze(golden_prompts, regular_prompts)
    report_path = os.path.join(results_dir, "pattern_report.json")
    miner.save_report(report, report_path)

    # Also rank all prompts
    ranker = PromptRanker()
    all_prompts = golden_prompts + regular_prompts
    ranked = ranker.rank(all_prompts)

    ranking_path = os.path.join(results_dir, "prompt_rankings.json")
    ranking_data = [
        {"rank": sp.rank, "score": sp.score, "prompt": sp.prompt[:300], "breakdown": sp.breakdown}
        for sp in ranked
    ]
    with open(ranking_path, "w", encoding="utf-8") as f:
        json.dump(ranking_data, f, indent=2, ensure_ascii=False)

    print(f"\n[Analysis] Pattern Report: {report_path}")
    print(f"[Analysis] Prompt Rankings: {ranking_path}")
    print(f"\n=== Top Insights ===")
    print(f"Golden Keywords: {', '.join(report.get('golden_keywords', [])[:8])}")
    print(f"Regular Keywords: {', '.join(report.get('regular_keywords', [])[:8])}")
    print(f"Summary: {report.get('summary', '')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Reverse Prompt Engineering for AI Educational Videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --video data/raw_videos/golden/neural_network.mp4 --group golden
  python main.py --batch data/raw_videos/golden/ --group golden --output results/golden/
  python main.py --download --max 5
  python main.py --analyze --results results/
        """,
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--video", help="Path to a single video file")
    input_group.add_argument("--batch", help="Directory of videos to process")

    # Data download
    parser.add_argument("--download", action="store_true", help="Download video dataset first")
    parser.add_argument("--max", type=int, default=5, help="Max videos per group to download (default: 5)")

    # Processing options
    parser.add_argument("--group", choices=["golden", "regular"], default="golden",
                        help="Video group label for analysis (default: golden)")
    parser.add_argument("--lang", choices=["en", "zh"], default="en",
                        help="Language for VLM prompts (default: en)")
    parser.add_argument("--output", default="results",
                        help="Output directory for prompt JSON files (default: results/)")

    # Analysis
    parser.add_argument("--analyze", action="store_true",
                        help="Run pattern mining on existing results")
    parser.add_argument("--results", default="results",
                        help="Results directory for pattern analysis (default: results/)")

    args = parser.parse_args()

    # Timestamp for run tracking
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n=== Reverse Prompt Engineering Pipeline ===")
    print(f"Run ID: {run_id}")

    # --- Step 0: Download data if requested ---
    if args.download:
        from src.utils.data_downloader import download_dataset
        print("\n[Step 0] Downloading video dataset...")
        download_dataset(output_dir="data/raw_videos", group="all", max_videos=args.max)

    # --- Step 1/2: Process video(s) ---
    if args.video:
        run_single_video(
            video_path=args.video,
            output_dir=args.output,
            group=args.group,
            lang=args.lang,
        )
    elif args.batch:
        run_batch(
            video_dir=args.batch,
            output_dir=args.output,
            group=args.group,
            lang=args.lang,
        )

    # --- Step 3: Pattern analysis ---
    if args.analyze:
        run_pattern_analysis(args.results)

    if not any([args.video, args.batch, args.download, args.analyze]):
        parser.print_help()
        print("\n[INFO] No action specified. See examples above.")


if __name__ == "__main__":
    main()
