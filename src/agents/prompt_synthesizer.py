"""
Prompt Synthesizer - Assembles Final Reversed Prompt from Agent Outputs
========================================================================
Takes outputs from all three agents (Pedagogy, Art Director, Structure)
and VLM analysis to construct a complete, structured "reverse-engineered" prompt.

The output format matches what AI video generation tools (Sora, Pika, 可灵, Runway)
typically expect: Subject + Action + Style + Camera + Quality + Negative.

Usage:
    synthesizer = PromptSynthesizer()
    result = synthesizer.synthesize(pedagogy_intent, art_style, structure, vlm_output)
"""

import json
from datetime import datetime


class PromptSynthesizer:
    """
    Assembles the final reverse-engineered prompt from multi-agent outputs.
    No model loading required - pure string assembly logic.
    """

    def __init__(self):
        pass

    def synthesize(
        self,
        pedagogy_intent: str,
        art_style: str,
        structure: str,
        vlm_description: dict | str,
        source_frame: str = "",
    ) -> dict:
        """
        Synthesize a complete reversed prompt from all agent outputs.
        
        Args:
            pedagogy_intent: Output from PedagogyAgent.
            art_style: Output from ArtDirectorAgent.
            structure: Output from StructureAgent.
            vlm_description: Original VLM analysis dict or string.
            source_frame: Path to the source keyframe (for traceability).
        
        Returns:
            dict with keys:
              - "full_prompt": Complete formatted prompt string
              - "structured": Dict with per-section breakdown
              - "metadata": Traceability info
        """
        # Extract core fields from VLM output
        if isinstance(vlm_description, dict):
            subject = vlm_description.get("subject", "educational content")
            action = vlm_description.get("action", "")
            text_overlays = vlm_description.get("text_overlays", "")
            scaffolding = vlm_description.get("scaffolding_cues", "")
            quality = vlm_description.get("overall_quality", "high")
        else:
            subject = str(vlm_description)[:100]
            action = ""
            text_overlays = ""
            scaffolding = ""
            quality = "high"

        # Map quality level to prompt keywords
        quality_map = {
            "cinematic": "cinematic quality, ultra-detailed, professional production",
            "high": "high quality, detailed, professional",
            "medium": "clean visuals, good quality",
            "low": "basic quality",
        }
        quality_kw = quality_map.get(quality, "high quality")

        # Build the structured sections
        structured = {
            "subject_and_action": self._build_subject_section(subject, action),
            "visual_style": art_style.strip(),
            "pedagogical_context": self._build_pedagogy_section(
                pedagogy_intent, scaffolding, text_overlays
            ),
            "camera_and_composition": structure.strip(),
            "quality_tags": quality_kw,
            "negative_prompt": self._build_negative_prompt(quality),
        }

        # Assemble the full prompt
        full_prompt = self._assemble_prompt(structured)

        return {
            "full_prompt": full_prompt,
            "structured": structured,
            "metadata": {
                "source_frame": source_frame,
                "generated_at": datetime.now().isoformat(),
                "version": "1.0",
            },
        }

    def _build_subject_section(self, subject: str, action: str) -> str:
        if action and action != "unknown":
            return f"{subject}, {action}"
        return subject

    def _build_pedagogy_section(
        self, pedagogy_intent: str, scaffolding: str, text_overlays: str
    ) -> str:
        parts = []
        if scaffolding and scaffolding != "unknown":
            parts.append(f"with {scaffolding}")
        if text_overlays and text_overlays != "unknown" and text_overlays != "none":
            parts.append(f"text labels: {text_overlays}")
        scaffold_str = ", ".join(parts)
        return f"{pedagogy_intent.split('.')[0]}. {scaffold_str}".strip(". ")

    def _build_negative_prompt(self, quality: str) -> str:
        base_negatives = "blurry, low quality, watermark, text errors, distorted"
        if quality == "cinematic":
            return f"{base_negatives}, amateur lighting, noisy, pixelated"
        return base_negatives

    def _assemble_prompt(self, structured: dict) -> str:
        """
        Assembles all sections into a single prompt string.
        Format inspired by Sora/Runway style prompts.
        """
        sections = [
            structured["subject_and_action"],
            structured["visual_style"],
            structured["camera_and_composition"],
            structured["quality_tags"],
        ]
        # Filter empty sections
        main_prompt = ", ".join([s for s in sections if s and s.strip()])
        
        # Add negative prompt as a separate annotation
        full = f"{main_prompt}.\n[Pedagogical Context]: {structured['pedagogical_context']}"
        if structured.get("negative_prompt"):
            full += f"\n[Negative]: {structured['negative_prompt']}"
        
        return full

    def save_result(self, result: dict, output_path: str):
        """Save a synthesized prompt result to a JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[Synthesizer] Saved to: {output_path}")


if __name__ == "__main__":
    # Test with sample agent outputs
    sample_vlm = {
        "subject": "DNA double helix structure",
        "action": "slowly rotating to reveal base pair bonds",
        "visual_style": "Cinematic 3D CGI animation",
        "lighting": "Volumetric blue ambient lighting",
        "camera_angle": "Close-up orbital shot",
        "color_palette": "Deep blue, white, teal",
        "scaffolding_cues": "color-coded base pairs, rotation arrows",
        "text_overlays": "Adenine, Thymine, base pair labels",
        "overall_quality": "cinematic",
    }
    
    pedagogy = (
        "Teaching Intent: Emphasis on structural relationships in molecular biology. "
        "Strategy: Visual Scaffolding via color-coding and rotational revelation."
    )
    art = "cinematic 3D animation, volumetric blue lighting, photorealistic CGI, shallow depth of field, 8K"
    structure = "Slow orbital camera movement, centered subject composition, gradual rotation reveal."
    
    synthesizer = PromptSynthesizer()
    result = synthesizer.synthesize(pedagogy, art, structure, sample_vlm, "scene_001.jpg")
    
    print("=== Final Reversed Prompt ===")
    print(result["full_prompt"])
    print("\n=== Structured Breakdown ===")
    print(json.dumps(result["structured"], indent=2, ensure_ascii=False))
