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
    _device = None  # Cache device to avoid querying parameters

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
            
            # Determine target device
            if device == "auto":
                target_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                target_device = device
            
            # Choose dtype based on device
            if target_device == "cuda":
                dtype = torch.float16  # Use float16 for T4 compatibility
            else:
                dtype = torch.float32
            
            # Load model WITHOUT device_map to avoid potential meta tensor issues
            cls._model = AutoModelForCausalLM.from_pretrained(
                cls.MODEL_ID,
                torch_dtype=dtype,
                trust_remote_code=True,
            ).to(target_device).eval()
            
            # Ensure GenerationMixin is available (for transformers>=4.50 compatibility)
            from transformers.generation.utils import GenerationMixin
            if not isinstance(cls._model, GenerationMixin):
                # If model doesn't have GenerationMixin, add it to class hierarchy
                model_class = cls._model.__class__
                if not issubclass(model_class, GenerationMixin):
                    model_class.__bases__ = (GenerationMixin,) + model_class.__bases__
                    print(f"[LLMAgent] Added GenerationMixin to {model_class.__name__}")
            
            # Cache device to avoid querying parameters
            cls._device = target_device
            print(f"[LLMAgent] Model loaded on: {target_device}")

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
        ).to(LLMAgentBase._device)
        
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
