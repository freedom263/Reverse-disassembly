"""
Pedagogy Agent - Infers Teaching Intent from Visual Analysis
=============================================================
Given VLM frame analysis, infers the pedagogical strategy and educational
intent behind the visual design. Uses Qwen2.5-1.5B-Instruct locally.

Usage:
    from src.agents.pedagogy_agent import PedagogyAgent
    agent = PedagogyAgent()
    intent = agent.analyze(vlm_output)
"""

from .llm_base import LLMAgentBase

SYSTEM_PROMPT = """You are an expert in educational technology and instructional design.
Your task is to analyze a visual description of an AI-generated educational video frame
and infer the pedagogical intent and teaching strategy used by the video creator.

Focus on:
- What teaching strategy is being used (Direct Instruction, Visual Cueing, Scaffolding, etc.)
- What cognitive goals is the video aiming to achieve
- What learning theory underpins the design choices (e.g., Dual Coding, Cognitive Load Theory)

Be concise and specific. Output in English, 2-3 sentences maximum."""


class PedagogyAgent(LLMAgentBase):
    """
    Infers pedagogical intent from VLM frame analysis.
    Shares the same Qwen2.5-1.5B model instance as other agents.
    """

    def __init__(self, device: str = "auto"):
        super().__init__(device=device)

    def analyze(self, vlm_description: dict | str) -> str:
        """
        Analyze the pedagogical intent from a VLM frame description.
        
        Args:
            vlm_description: Dict from VLMAnalyzer.analyze_keyframe(), or a raw string.
        
        Returns:
            str: Pedagogical intent description (suitable for prompt injection).
        
        Example output:
            "Teaching Intent: Emphasis on structural relationships (Concept Mapping);
             Strategy: Visual scaffolding with directional cues. Cognitive goal: 
             Reduce extraneous load by isolating key anatomical features."
        """
        # Format input
        if isinstance(vlm_description, dict):
            lines = []
            for k, v in vlm_description.items():
                if not k.startswith("_") and v and v != "unknown":
                    lines.append(f"- {k}: {v}")
            description_text = "\n".join(lines)
        else:
            description_text = str(vlm_description)
        
        user_message = (
            f"This is an AI-generated educational video frame analysis:\n\n"
            f"{description_text}\n\n"
            f"What is the pedagogical intent and teaching strategy reflected "
            f"in this visual design? How does it serve the learner's cognitive process?"
        )
        
        response = self._chat(
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
            max_new_tokens=200,
            temperature=0.3,
        )
        return response


if __name__ == "__main__":
    # Quick test with a sample VLM output
    sample_vlm = {
        "subject": "DNA double helix structure",
        "action": "Rotation revealing base pairs with color coding",
        "visual_style": "Cinematic 3D CGI animation",
        "lighting": "Dramatic side lighting with blue glow",
        "scaffolding_cues": "Color-coded base pairs, rotation arrows, label callouts",
        "overall_quality": "cinematic",
    }
    
    agent = PedagogyAgent()
    result = agent.analyze(sample_vlm)
    print("=== Pedagogy Agent Output ===")
    print(result)
