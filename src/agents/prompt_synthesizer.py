class PromptSynthesizer:
    def __init__(self):
        pass

    def synthesize(self, pedagogy_intent, art_style, vlm_description):
        """
        Synthesizes the final AIGC prompt from agent outputs.
        
        Args:
            pedagogy_intent (str): Output from Pedagogy Agent.
            art_style (str): Output from Art Director Agent.
            vlm_description (dict/str): Output from VLM Analyzer.
            
        Returns:
            str: The synthesized prompt.
        """
        # A basic template combining the elements
        
        subject_action = ""
        if isinstance(vlm_description, dict):
             visuals = ", ".join(vlm_description.get("visual_elements", []))
             content = ", ".join(vlm_description.get("text_content", []))
             subject_action = f"Subject shows {content}. Action/Visuals include {visuals}."
        
        prompt = (
            f"Generate a high-quality educational video scene.\n"
            f"[{art_style}]\n"
            f"[Subject & Action]: {subject_action}\n"
            f"[Pedagogical Intent]: {pedagogy_intent}\n"
            f"[Camera/Lighting]: Well-lit, educational framing, clear focus on the subject."
        )
        return prompt
