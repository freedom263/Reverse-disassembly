"""
LLM Agent Base - Local Qwen2.5-1.5B-Instruct Inference
==========================================================
Shared base class for all reasoning agents. Loads Qwen2.5-1.5B-Instruct once
and reuses it across all agents to conserve GPU VRAM on T4.

Model: Qwen/Qwen2.5-1.5B-Instruct (~3GB VRAM)
Supports: Chinese & English

Usage: Do not use directly; import via PedagogyAgent, ArtDirectorAgent, etc.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLMAgentBase:
    """
    Base class providing shared Qwen2.5-1.5B-Instruct inference.
    Uses singleton pattern so the model is loaded only once.
    """

    MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
    _model = None
    _tokenizer = None

    @classmethod
    def _ensure_loaded(cls, device: str = "auto"):
        """Load model once; subsequent calls are no-ops."""
        if cls._model is None:
            print(f"[LLMAgent] Loading model: {cls.MODEL_ID}")
            print("  (First run will download ~3GB weights from HuggingFace)")
            cls._tokenizer = AutoTokenizer.from_pretrained(
                cls.MODEL_ID,
                trust_remote_code=True,
            )
            cls._model = AutoModelForCausalLM.from_pretrained(
                cls.MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map=device,
                trust_remote_code=True,
            ).eval()
            print(f"[LLMAgent] Model loaded on: {next(cls._model.parameters()).device}")

    def __init__(self, device: str = "auto"):
        LLMAgentBase._ensure_loaded(device)
        self.model = LLMAgentBase._model
        self.tokenizer = LLMAgentBase._tokenizer

    def _chat(
        self,
        system_prompt: str,
        user_message: str,
        max_new_tokens: int = 300,
        temperature: float = 0.3,
    ) -> str:
        """
        Run a chat-completion style inference.
        
        Args:
            system_prompt: System role instruction for the agent.
            user_message: The actual query.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (lower = more deterministic).
        
        Returns:
            str: Generated response text.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt",
        ).to(next(self.model.parameters()).device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the newly generated tokens
        new_tokens = generated_ids[0][model_inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()
