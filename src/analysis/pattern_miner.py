"""
Pattern Miner - Real TF-IDF + Embedding Clustering Analysis
=============================================================
Analyzes a collection of reversed prompts to identify patterns that
distinguish "Golden" (high-quality) prompts from "Regular" ones.

Methods:
1. TF-IDF: Find keywords uniquely frequent in Golden group
2. Sentence Embeddings: Cluster prompts to find recurring themes
3. Keyword Frequency: Simple frequency analysis with group comparison

No GPU required - runs on CPU using sentence-transformers.

Usage:
    from src.analysis.pattern_miner import PatternMiner
    miner = PatternMiner()
    report = miner.analyze(golden_prompts, regular_prompts)
    miner.save_report(report, "results/pattern_report.json")
"""

import json
import os
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class PatternMiner:
    """
    Analyzes reversed prompts to extract 'Golden Patterns' -
    keywords and phrases that characterize high-quality AI video prompts.
    """

    EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, embed_model: str = None):
        """
        Args:
            embed_model: SentenceTransformer model ID. Defaults to multilingual MiniLM.
                         (Supports Chinese + English, ~90MB)
        """
        self.embed_model_id = embed_model or self.EMBED_MODEL
        print(f"[PatternMiner] Loading embedding model: {self.embed_model_id}")
        self.embedder = SentenceTransformer(self.embed_model_id)
        print("[PatternMiner] Ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        golden_prompts: list[str],
        regular_prompts: list[str],
        n_clusters: int = 5,
        top_n_keywords: int = 20,
    ) -> dict:
        """
        Full analysis pipeline comparing Golden vs Regular prompts.
        
        Args:
            golden_prompts: List of reversed prompts from high-quality videos.
            regular_prompts: List of reversed prompts from regular videos.
            n_clusters: Number of clusters for KMeans on Golden prompts.
            top_n_keywords: Number of differentiating keywords to report.
        
        Returns:
            dict with keys:
              - "golden_keywords": Top TF-IDF keywords unique to Golden group
              - "regular_keywords": Top TF-IDF keywords unique to Regular group
              - "golden_clusters": Cluster themes found in Golden prompts
              - "keyword_frequency": Frequency comparison table
              - "summary": Human-readable summary
        """
        print(f"\n[PatternMiner] Analyzing {len(golden_prompts)} golden vs "
              f"{len(regular_prompts)} regular prompts...")
        
        all_prompts = golden_prompts + regular_prompts
        labels = ["golden"] * len(golden_prompts) + ["regular"] * len(regular_prompts)
        
        if len(all_prompts) < 2:
            return {"error": "Need at least 2 prompts total for analysis."}
        
        # 1. TF-IDF keyword extraction
        tfidf_results = self._tfidf_analysis(
            golden_prompts, regular_prompts, top_n_keywords
        )
        
        # 2. Embedding clustering on Golden group
        cluster_results = {}
        if len(golden_prompts) >= n_clusters:
            cluster_results = self._cluster_golden_prompts(
                golden_prompts, n_clusters
            )
        else:
            print(f"  [SKIP] Clustering skipped: need >= {n_clusters} golden prompts")
        
        # 3. Frequency comparison
        freq_comparison = self._keyword_frequency_comparison(
            golden_prompts, regular_prompts, top_n=top_n_keywords
        )
        
        report = {
            "golden_keywords": tfidf_results["golden_exclusive"],
            "regular_keywords": tfidf_results["regular_exclusive"],
            "golden_clusters": cluster_results,
            "keyword_frequency": freq_comparison,
            "summary": self._generate_summary(tfidf_results, cluster_results),
            "stats": {
                "golden_count": len(golden_prompts),
                "regular_count": len(regular_prompts),
                "avg_golden_length": int(np.mean([len(p.split()) for p in golden_prompts])) if golden_prompts else 0,
                "avg_regular_length": int(np.mean([len(p.split()) for p in regular_prompts])) if regular_prompts else 0,
            },
        }
        
        return report

    def load_prompts_from_results(self, results_dir: str) -> tuple[list[str], list[str]]:
        """
        Load reversed prompts from JSON files in a results directory.
        
        Expects files named: <video_title>_prompt.json
        Each JSON has a "full_prompt" key and a "metadata.group" key.
        
        Args:
            results_dir: Path to directory containing *_prompt.json files.
        
        Returns:
            (golden_prompts, regular_prompts) tuple of string lists.
        """
        golden, regular = [], []
        
        if not os.path.exists(results_dir):
            print(f"[PatternMiner] Results directory not found: {results_dir}")
            return golden, regular
        
        for fname in os.listdir(results_dir):
            if not fname.endswith("_prompt.json"):
                continue
            fpath = os.path.join(results_dir, fname)
            try:
                with open(fpath, encoding="utf-8") as f:
                    data = json.load(f)
                prompt = data.get("full_prompt", "")
                group = data.get("metadata", {}).get("group", "regular")
                if not prompt:
                    continue
                if group == "golden":
                    golden.append(prompt)
                else:
                    regular.append(prompt)
            except Exception as e:
                print(f"  [WARN] Could not load {fname}: {e}")
        
        print(f"[PatternMiner] Loaded {len(golden)} golden + {len(regular)} regular prompts")
        return golden, regular

    def save_report(self, report: dict, output_path: str):
        """Save the analysis report to JSON."""
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[PatternMiner] Report saved to: {output_path}")

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _tfidf_analysis(
        self,
        golden_prompts: list[str],
        regular_prompts: list[str],
        top_n: int,
    ) -> dict:
        """Use TF-IDF to find keywords exclusive to each group."""
        if not golden_prompts or not regular_prompts:
            return {"golden_exclusive": [], "regular_exclusive": []}
        
        # Combine each group into a single 'document' for group-level TF-IDF
        golden_doc = " ".join(golden_prompts).lower()
        regular_doc = " ".join(regular_prompts).lower()
        
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),    # unigrams, bigrams, trigrams
            min_df=1,
            max_features=500,
            stop_words="english",
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform([golden_doc, regular_doc])
            feature_names = np.array(vectorizer.get_feature_names_out())
            
            golden_scores = tfidf_matrix[0].toarray().flatten()
            regular_scores = tfidf_matrix[1].toarray().flatten()
            
            # Terms with high score in Golden but low in Regular
            diff_golden = golden_scores - regular_scores * 0.5
            diff_regular = regular_scores - golden_scores * 0.5
            
            golden_top_idx = diff_golden.argsort()[-top_n:][::-1]
            regular_top_idx = diff_regular.argsort()[-top_n:][::-1]
            
            return {
                "golden_exclusive": feature_names[golden_top_idx].tolist(),
                "regular_exclusive": feature_names[regular_top_idx].tolist(),
            }
        except Exception as e:
            print(f"  [WARN] TF-IDF failed: {e}")
            return {"golden_exclusive": [], "regular_exclusive": []}

    def _cluster_golden_prompts(
        self, golden_prompts: list[str], n_clusters: int
    ) -> dict:
        """Cluster Golden prompts by semantic similarity."""
        print(f"  Encoding {len(golden_prompts)} golden prompts for clustering...")
        embeddings = self.embedder.encode(golden_prompts, show_progress_bar=False)
        
        n_clusters = min(n_clusters, len(golden_prompts))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(embeddings)
        
        clusters = {}
        for cluster_id in range(n_clusters):
            indices = np.where(cluster_labels == cluster_id)[0]
            cluster_prompts = [golden_prompts[i] for i in indices]
            
            # Find the representative (closest to centroid)
            centroid = kmeans.cluster_centers_[cluster_id]
            cluster_embeddings = embeddings[indices]
            sims = cosine_similarity([centroid], cluster_embeddings)[0]
            rep_idx = indices[sims.argmax()]
            
            clusters[f"cluster_{cluster_id}"] = {
                "size": len(indices),
                "representative": golden_prompts[rep_idx][:200],
                "sample_prompts": [p[:150] for p in cluster_prompts[:3]],
            }
        
        return clusters

    def _keyword_frequency_comparison(
        self,
        golden_prompts: list[str],
        regular_prompts: list[str],
        top_n: int,
    ) -> list[dict]:
        """Compare keyword frequencies between groups."""
        def tokenize(prompts):
            text = " ".join(prompts).lower()
            # Keep only meaningful words (length >= 3)
            words = re.findall(r'\b[a-z]{3,}\b', text)
            stopwords = {
                "the", "and", "for", "with", "this", "that", "are",
                "is", "in", "of", "to", "a", "an", "it", "its",
                "from", "by", "as", "on", "at", "be", "has", "have"
            }
            return [w for w in words if w not in stopwords]
        
        golden_words = tokenize(golden_prompts)
        regular_words = tokenize(regular_prompts)
        
        golden_freq = Counter(golden_words)
        regular_freq = Counter(regular_words)
        
        # Normalize by group size
        total_g = max(len(golden_words), 1)
        total_r = max(len(regular_words), 1)
        
        all_keywords = set(list(golden_freq.keys())[:100] + list(regular_freq.keys())[:100])
        
        comparison = []
        for kw in all_keywords:
            g_rate = golden_freq[kw] / total_g
            r_rate = regular_freq[kw] / total_r
            if g_rate + r_rate > 0:
                comparison.append({
                    "keyword": kw,
                    "golden_pct": round(g_rate * 100, 2),
                    "regular_pct": round(r_rate * 100, 2),
                    "golden_advantage": round((g_rate - r_rate) * 100, 2),
                })
        
        # Sort by how much more common in Golden vs Regular
        comparison.sort(key=lambda x: x["golden_advantage"], reverse=True)
        return comparison[:top_n]

    def _generate_summary(self, tfidf_results: dict, cluster_results: dict) -> str:
        top_golden = tfidf_results.get("golden_exclusive", [])[:5]
        n_clusters = len(cluster_results)
        summary = (
            f"Analysis complete. Top distinguishing keywords in Golden prompts: "
            f"{', '.join(top_golden) if top_golden else 'N/A'}. "
            f"Golden prompts organized into {n_clusters} semantic clusters."
        )
        return summary


if __name__ == "__main__":
    # Test with synthetic data
    golden_sample = [
        "DNA double helix, slowly rotating orbital shot, cinematic 3D CGI, volumetric blue lighting, "
        "photorealistic, 8K, shallow depth of field, color-coded base pairs, scientific visualization",
        "Neural network layers, animated data flow, motion graphics style, soft gradient background, "
        "professional animation, dynamic transitions, high contrast colors, cinematic quality",
        "Photosynthesis process inside chloroplast, macro close-up, bioluminescent particles, "
        "cinematic documentary style, warm golden lighting, slow reveal, 4K quality",
    ]
    regular_sample = [
        "Teacher explaining on whiteboard, talking head shot, basic lighting, screen recording style",
        "Slides with text and diagrams, static shot, plain white background, simple animation",
        "Classroom scene with students, wide shot, natural lighting, documentary style",
    ]
    
    miner = PatternMiner()
    report = miner.analyze(golden_sample, regular_sample)
    
    print("\n=== Pattern Mining Report ===")
    print(f"Golden keywords: {report['golden_keywords'][:10]}")
    print(f"Regular keywords: {report['regular_keywords'][:10]}")
    print(f"\nSummary: {report['summary']}")
