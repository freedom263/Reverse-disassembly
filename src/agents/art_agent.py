"""
Art Director Agent - Extracts Visual Style Parameters for Prompt Reconstruction
================================================================================
Given VLM frame analysis, extracts precise aesthetic parameters that can be
directly used as prompt keywords for AI video generation tools (Sora, Pika, 可灵).

Usage:
    from src.agents.art_agent import ArtDirectorAgent
    agent = ArtDirectorAgent()
    style_params = agent.analyze(vlm_output)
"""

from .llm_base import LLMAgentBase

SYSTEM_PROMPT = """You are a professional AI video production prompt engineer and art director.
Your task is to convert a visual analysis of an AI-generated video frame into precise,
actionable style keywords that can be used in AI video generation prompts.

Output format: Include specific technical terms used in AI video prompts such as:
- Lighting: "soft rim lighting", "volumetric lighting", "neon glow", "studio lighting"
- Quality: "8K", "photorealistic", "hyper-detailed", "cinematic quality"
- Style: "Pixar style", "motion graphics", "documentary style", "scientific visualization"
- Camera: "macro lens", "wide angle", "shallow depth of field", "tracking shot"
- Atmosphere: "ethereal", "clinical", "warm", "futuristic"

Output only the style parameters as a comma-separated list of English prompt keywords.
Maximum 30 words."""


class ArtDirectorAgent(LLMAgentBase):
    """
    Extracts aesthetic/style parameters from VLM analysis for prompt reconstruction.
    Shares the same Qwen2.5-1.5B model instance as other agents.
    """

    def __init__(self, device: str = "auto"):
        super().__init__(device=device)

    def analyze(self, vlm_description: dict | str) -> str:
        """
        Extract style parameters suitable for use in AI video generation prompts.
        
        Args:
            vlm_description: Dict from VLMAnalyzer.analyze_keyframe(), or a raw string.
        
        Returns:
            str: Comma-separated style keywords for prompt construction.
        
        Example output:
            "cinematic 3D animation, volumetric blue lighting, macro close-up,
             photorealistic CGI, shallow depth of field, scientific visualization, 8K"
        """
        if isinstance(vlm_description, dict):
            style_keys = ["visual_style", "lighting", "camera_angle", "color_palette",
                          "background", "overall_quality"]
            lines = []
            for k in style_keys:
                v = vlm_description.get(k, "")
                if v and v != "unknown":
                    lines.append(f"- {k}: {v}")
            description_text = "\n".join(lines) if lines else str(vlm_description)
        else:
            description_text = str(vlm_description)
        
        user_message = (
            f"Convert this AI video frame visual analysis into AI generation prompt keywords:\n\n"
            f"{description_text}\n\n"
            f"Output the style parameters as prompt keywords only (comma-separated)."
        )
        
        response = self._chat(
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
            max_new_tokens=100,
            temperature=0.2,
        )
        return response


if __name__ == "__main__":
    sample_vlm = {
        "visual_style": "Cinematic 3D CGI animation",
        "lighting": "Dramatic side lighting with blue glow",
        "camera_angle": "close-up rotating shot",
        "color_palette": "deep blue, white, teal accents",
        "background": "dark gradient with particle effects",
        "overall_quality": "cinematic",
    }
    
    agent = ArtDirectorAgent()
    result = agent.analyze(sample_vlm)
    print("=== Art Director Agent Output ===")
    print(result)
