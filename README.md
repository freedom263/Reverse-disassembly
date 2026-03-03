# 向上溯源反推系统 - Reverse Prompt Engineering for AI Educational Videos

## 项目概述 (Project Overview)

This system analyzes AI-generated educational videos to **reverse-engineer the prompts** that created them. It identifies what makes "Golden" (high-quality, cinematic) AI video prompts different from ordinary ones.

## 核心流程 (Pipeline)

```
Video (.mp4)
    ↓ [scene detection + keyframe extraction]
Keyframes (.jpg)
    ↓ [InternVL2-2B local inference]
VLM Analysis (JSON: subject, style, lighting, camera...)
    ↓ [3 parallel LLM Agents - Qwen2.5-1.5B]
    ├── Pedagogy Agent → teaching intent
    ├── Art Director Agent → visual style keywords  
    └── Structure Agent → camera & composition
    ↓ [Prompt Synthesizer]
Reversed Prompt (structured JSON + full prompt string)
    ↓ [Pattern Miner]
Golden Pattern Report (differentiating keywords + clusters)
```

## 快速开始 (Quick Start)

### Colab (推荐)
Open `colab_run.ipynb` in Google Colab with T4 GPU, run all cells.

### 本地开发 (Local Dev - no GPU needed for testing logic)
```bash
# Install dependencies
pip install -r requirements.txt

# Download a small sample dataset
python main.py --download --max 2

# Process golden videos
python main.py --batch data/raw_videos/golden/ --group golden --output results/golden/

# Pattern analysis
python main.py --analyze --results results/
```

## 项目结构 (Structure)

```
├── main.py                         # CLI entry point
├── colab_run.ipynb                 # Colab end-to-end notebook
├── requirements.txt                # Dependencies
├── src/
│   ├── utils/
│   │   ├── video_processor.py      # Scene detection + keyframe extraction
│   │   └── data_downloader.py      # yt-dlp based video downloader
│   ├── perception/
│   │   └── vlm_analyzer.py         # InternVL2-2B frame analysis
│   ├── agents/
│   │   ├── llm_base.py             # Shared Qwen2.5-1.5B base class
│   │   ├── pedagogy_agent.py       # Teaching intent inference
│   │   ├── art_agent.py            # Visual style extraction
│   │   ├── structure_agent.py      # Camera/composition inference
│   │   └── prompt_synthesizer.py   # Final prompt assembly
│   └── analysis/
│       ├── pattern_miner.py        # TF-IDF + embedding clustering
│       └── prompt_ranker.py        # Rule-based prompt quality scoring
└── data/
    └── raw_videos/
        ├── golden/                 # High-quality AIGC educational videos
        └── regular/                # Standard educational videos
```

## 模型选择 (Models)

| Component | Model | VRAM | Purpose |
|-----------|-------|------|---------|
| VLM | InternVL2-2B | ~4GB | Frame visual analysis |
| LLM Agents | Qwen2.5-1.5B-Instruct | ~3GB | Reasoning & text gen |
| Embeddings | MiniLM-L12-v2 | CPU | Pattern clustering |
| **Total** | | **~10GB** | Well within T4 (15GB) |

## 数据说明 (Data)

- **Golden Group**: 3Blue1Brown, Kurzgesagt — cinematic, high-production educational videos
- **Regular Group**: Standard lectures, screen recordings, talking-head style
- Videos are downloaded via `yt-dlp` (≤720p, ≤10min to keep size manageable)
