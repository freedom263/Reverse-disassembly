"""
Prompt Ranker - Score Reversed Prompts for Specificity & Quality
================================================================
After reverse engineering, ranks the retrieved prompts by how "golden"
they are - i.e., how specific, rich, and cinematically diverse.

Scoring criteria:
1. Length / richness: longer, more detailed prompts score higher
2. Technical vocabulary: presence of AI video generation keywords
3. Specificity: avoids generic terms, uses concrete descriptors
4. Diversity: covers multiple prompt dimensions (style, camera, light, etc.)

No GPU needed - runs on CPU.

Usage:
    from src.analysis.prompt_ranker import PromptRanker
    ranker = PromptRanker()
    scored = ranker.rank(list_of_prompts)
"""

import re
from dataclasses import dataclass


# Technical AI video generation vocabulary (high-value keywords)
GOLDEN_VOCABULARY = {
    # Quality / Resolution
    "cinematic", "8k", "4k", "ultra-detailed", "hyper-detailed",
    "photorealistic", "physically-based rendering", "ray tracing",
    # Lighting
    "volumetric", "rim lighting", "studio lighting", "golden hour",
    "bioluminescent", "neon glow", "chiaroscuro", "dramatic lighting",
    # Style
    "motion graphics", "scientific visualization", "cgi animation",
    "3d render", "exploded view", "macro", "slow motion",
    # Camera
    "orbital shot", "tracking shot", "shallow depth of field",
    "bokeh", "aerial", "time-lapse", "dolly zoom",
    # Production
    "professional production", "award-winning", "high production value",
    # Educational specifics
    "color-coded", "annotated", "labelled", "scaffolding", "callout",
}

GENERIC_PENALTY_WORDS = {
    "good", "nice", "basic", "simple", "plain", "normal",
    "standard", "regular", "ordinary", "common",
}

PROMPT_DIMENSIONS = [
    "subject", "action", "style", "lighting", "camera",
    "color", "background", "quality", "motion", "composition",
]

DIMENSION_KEYWORDS = {
    "style": ["animation", "cgi", "3d", "motion", "cinematic", "render"],
    "lighting": ["lighting", "light", "illuminat", "glow", "shadow", "bright"],
    "camera": ["shot", "angle", "view", "zoom", "pan", "close-up", "wide"],
    "color": ["color", "colour", "palette", "hue", "tone", "gradient"],
    "quality": ["8k", "4k", "quality", "detailed", "resolution", "hd"],
    "motion": ["motion", "moving", "rotating", "flowing", "animated", "dynamic"],
}


@dataclass
class ScoredPrompt:
    prompt: str
    score: float
    breakdown: dict
    rank: int = 0


class PromptRanker:
    """
    Scores and ranks reversed prompts by quality and specificity.
    """

    def __init__(self):
        pass

    def score(self, prompt: str) -> ScoredPrompt:
        """
        Score a single prompt.
        
        Args:
            prompt: The reversed prompt string.
        
        Returns:
            ScoredPrompt with score (0.0-100.0) and breakdown.
        """
        prompt_lower = prompt.lower()
        words = set(re.findall(r'\b\w+\b', prompt_lower))
        word_count = len(prompt.split())
        
        # --- Scoring components ---
        
        # 1. Length richness (0-25 pts): max at ~50 words
        length_score = min(word_count / 50, 1.0) * 25
        
        # 2. Golden vocabulary hits (0-30 pts)
        vocab_hits = sum(
            1 for kw in GOLDEN_VOCABULARY
            if kw in prompt_lower
        )
        vocab_score = min(vocab_hits / 8, 1.0) * 30
        
        # 3. Dimension coverage (0-25 pts): how many aspects are covered
        dim_covered = 0
        for dim, keywords in DIMENSION_KEYWORDS.items():
            if any(kw in prompt_lower for kw in keywords):
                dim_covered += 1
        dim_score = (dim_covered / len(DIMENSION_KEYWORDS)) * 25
        
        # 4. Specificity: penalize generic words (0-10 pts)
        generic_hits = sum(1 for w in words if w in GENERIC_PENALTY_WORDS)
        specificity_score = max(10 - generic_hits * 3, 0)
        
        # 5. Has pedagogical context section (0-10 pts)
        pedagogy_score = 10 if "[pedagogical context]" in prompt_lower else 0
        
        total = length_score + vocab_score + dim_score + specificity_score + pedagogy_score
        
        return ScoredPrompt(
            prompt=prompt,
            score=round(total, 2),
            breakdown={
                "length_richness": round(length_score, 2),
                "golden_vocabulary": round(vocab_score, 2),
                "dimension_coverage": round(dim_score, 2),
                "specificity": round(specificity_score, 2),
                "pedagogical_context": pedagogy_score,
                "vocab_hits": vocab_hits,
                "dimensions_covered": dim_covered,
                "word_count": word_count,
            },
        )

    def rank(self, prompts: list[str]) -> list[ScoredPrompt]:
        """
        Score and rank a list of prompts.
        
        Args:
            prompts: List of prompt strings.
        
        Returns:
            List of ScoredPrompt, sorted by score descending.
        """
        scored = [self.score(p) for p in prompts]
        scored.sort(key=lambda x: x.score, reverse=True)
        for i, sp in enumerate(scored):
            sp.rank = i + 1
        return scored

    def get_golden_threshold(self, scored_prompts: list[ScoredPrompt]) -> float:
        """
        Estimate the score threshold to classify a prompt as 'Golden'.
        Uses the top-33% percentile as the cutoff.
        """
        if not scored_prompts:
            return 50.0
        scores = [sp.score for sp in scored_prompts]
        scores.sort(reverse=True)
        cutoff_idx = max(len(scores) // 3, 1)
        return scores[cutoff_idx - 1]


if __name__ == "__main__":
    test_prompts = [
        (
            "DNA double helix slowly rotating orbital shot, cinematic 3D CGI animation, "
            "volumetric blue lighting, photorealistic, 8K, shallow depth of field, "
            "color-coded base pairs, scientific visualization.\n"
            "[Pedagogical Context]: Emphasis on structural relationships with visual scaffolding."
        ),
        (
            "Neural network layers with data flow animation, motion graphics style, "
            "soft gradient background, professional animation, high contrast colors."
        ),
        "Teacher explaining on whiteboard, basic lighting, simple animation.",
        "Good video showing photosynthesis with normal style.",
    ]
    
    ranker = PromptRanker()
    ranked = ranker.rank(test_prompts)
    
    print("=== Prompt Rankings ===")
    for sp in ranked:
        print(f"\nRank #{sp.rank} | Score: {sp.score}/100")
        print(f"  Preview: {sp.prompt[:80]}...")
        print(f"  Breakdown: {sp.breakdown}")
