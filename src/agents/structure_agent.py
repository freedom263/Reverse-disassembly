"""
Structure Agent - Infers Composition & Camera Language for Prompt Reconstruction
==================================================================================
Given VLM frame analysis, infers the structural/compositional parameters:
shot type, camera motion, scene composition, pacing style.

These parameters help reconstruct the "camera/director" portion of an AI video prompt.

Usage:
    from src.agents.structure_agent import StructureAgent
    agent = StructureAgent()
    structure = agent.analyze(vlm_output)
"""

from .llm_base import LLMAgentBase

SYSTEM_PROMPT = """You are a professional cinematographer and AI video production specialist.
Your task is to analyze visual descriptions of AI-generated educational video frames and
infer the structural and cinematographic parameters used.

Focus on:
- Shot type (extreme close-up, close-up, medium, wide, aerial, etc.)
- Camera motion (static, slow zoom-in, pan, orbit/360, tracking)
- Scene composition (rule of thirds, centered, symmetrical, dynamic)
- Transition style (cut, dissolve, morph, zoom)
- Pacing (slow and deliberate, fast-paced, gradual reveal)

Output as a concise English description of 1-2 sentences, using specific cinematographic
terms suitable for inclusion in an AI video prompt."""


class StructureAgent(LLMAgentBase):
    """
    Infers cinematographic/compositional parameters from VLM analysis.
    """

    def __init__(self, device: str = "auto"):
        super().__init__(device=device)

    def analyze(self, vlm_description: dict | str) -> str:
        """
        Infer structural and cinematographic parameters for prompt reconstruction.
        
        Args:
            vlm_description: Dict from VLMAnalyzer or raw string.
        
        Returns:
            str: Camera/composition description for prompt inclusion.
        
        Example output:
            "Slow orbital camera movement around a centered subject, rule-of-thirds
             composition with shallow depth of field, gradual reveal with smooth zoom-in."
        """
        if isinstance(vlm_description, dict):
            keys = ["subject", "action", "camera_angle", "visual_style", "scaffolding_cues"]
            lines = []
            for k in keys:
                v = vlm_description.get(k, "")
                if v and v != "unknown":
                    lines.append(f"- {k}: {v}")
            description_text = "\n".join(lines) if lines else str(vlm_description)
        else:
            description_text = str(vlm_description)
        
        user_message = (
            f"Based on this AI educational video frame analysis:\n\n"
            f"{description_text}\n\n"
            f"Describe the cinematographic and compositional parameters "
            f"(shot type, camera motion, composition, pacing) in 1-2 sentences "
            f"using terms suitable for an AI video generation prompt."
        )
        
        response = self._chat(
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
            max_new_tokens=120,
            temperature=0.25,
        )
        return response


if __name__ == "__main__":
    sample_vlm = {
        "subject": "DNA double helix",
        "action": "Slowly rotating to reveal structure",
        "camera_angle": "close-up orbital shot",
        "visual_style": "3D CGI scientific visualization",
        "scaffolding_cues": "rotation arrows, highlighted base pairs",
    }
    
    agent = StructureAgent()
    result = agent.analyze(sample_vlm)
    print("=== Structure Agent Output ===")
    print(result)
