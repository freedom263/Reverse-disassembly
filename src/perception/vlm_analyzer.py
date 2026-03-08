"""
VLM Analyzer - Real Local Inference with InternVL2-2B
========================================================
Analyzes video keyframes using a local Vision-Language Model (VLM).
No API calls. Model weights downloaded from HuggingFace on first run.

Model: OpenGVLab/InternVL2-2B (~4GB VRAM on T4)
Supports: Chinese & English input/output

Usage:
    vlm = VLMAnalyzer()
    result = vlm.analyze_keyframe("data/keyframes/scene_001.jpg")
    # result: dict with visual_style, subject, action, camera_angle, etc.
"""

import json
import os
import re
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


# ---------------------------------------------------------------------------
# InternVL2 image preprocessing (official recipe)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def _build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def _load_image(image_path: str, input_size=448) -> torch.Tensor:
    """Load and preprocess image for InternVL2."""
    image = Image.open(image_path).convert("RGB")
    transform = _build_transform(input_size)
    pixel_values = transform(image).unsqueeze(0)  # (1, 3, H, W)
    return pixel_values


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

ANALYSIS_PROMPT_EN = """You are a professional AI video production analyst.
Analyze this educational video keyframe and extract information to reverse-engineer the prompt that generated it.

Output ONLY a valid JSON object with these exact keys:
{
  "subject": "main subject/object in the scene",
  "action": "what is happening / motion description",
  "visual_style": "art style (e.g. photorealistic, 3D animation, motion graphics, cartoon, cinematic)",
  "lighting": "lighting description (e.g. soft studio, dramatic side lighting, natural daylight)",
  "camera_angle": "camera angle (e.g. close-up, wide shot, bird's eye, eye-level)",
  "color_palette": "dominant colors (e.g. cool blues and whites, warm earth tones)",
  "background": "background description (e.g. clean white, abstract gradient, realistic environment)",
  "text_overlays": "any visible text or labels in the frame",
  "scaffolding_cues": "pedagogical visual aids present (e.g. arrows, highlights, diagrams, callouts)",
  "overall_quality": "production quality estimate (low/medium/high/cinematic)"
}
Do not include any explanation, only the JSON."""

ANALYSIS_PROMPT_ZH = """你是一位专业的AI视频制作分析师。
分析这张教学视频关键帧，提取信息以反推生成该视频的提示词。

仅输出以下格式的JSON，不要包含任何解释：
{
  "subject": "画面中的主要主体/对象",
  "action": "正在发生的事情/动作描述",
  "visual_style": "艺术风格（如：写实、3D动画、动态图形、卡通、电影级）",
  "lighting": "光照描述（如：柔和工作室灯光、戏剧性侧光、自然日光）",
  "camera_angle": "摄像角度（如：特写、广角、俯视、平视）",
  "color_palette": "主色调（如：冷蓝色和白色、温暖大地色）",
  "background": "背景描述（如：纯白、抽象渐变、真实环境）",
  "text_overlays": "画面中可见的文字或标签",
  "scaffolding_cues": "教学视觉辅助（如：箭头、高亮、图表、标注框）",
  "overall_quality": "制作质量估计（low/medium/high/cinematic）"
}"""

RETRY_PROMPT_EN = """Return ONLY one valid JSON object with keys:
subject, action, visual_style, lighting, camera_angle, color_palette, background, text_overlays, scaffolding_cues, overall_quality.
Use short phrases. If uncertain, use "unknown". No markdown. No extra text."""

RETRY_PROMPT_ZH = """仅输出一个有效 JSON 对象，键必须是：
subject, action, visual_style, lighting, camera_angle, color_palette, background, text_overlays, scaffolding_cues, overall_quality。
值尽量简短；不确定时填 "unknown"。不要 markdown，不要额外说明。"""


class VLMAnalyzer:
    """
    Analyzes video keyframes using InternVL2-2B running locally on GPU.
    
    On first instantiation, the model (~4GB) is downloaded from HuggingFace.
    Subsequent runs load from the local HuggingFace cache.
    """

    MODEL_ID = "OpenGVLab/InternVL2-2B"
    _instance_model = None
    _instance_tokenizer = None
    _instance_device = None
    _instance_dtype = None
    EXPECTED_KEYS = [
        "subject",
        "action",
        "visual_style",
        "lighting",
        "camera_angle",
        "color_palette",
        "background",
        "text_overlays",
        "scaffolding_cues",
        "overall_quality",
    ]

    def __init__(self, model_id: str = None, device: str = "auto", lang: str = "en"):
        """
        Args:
            model_id: HuggingFace model ID. Defaults to InternVL2-2B.
            device: "auto" (recommended for Colab), "cuda", or "cpu".
            lang: Prompt language "en" or "zh". Use "zh" for Chinese content.
        """
        self.model_id = model_id or self.MODEL_ID
        self.lang = lang
        self.prompt = ANALYSIS_PROMPT_ZH if lang == "zh" else ANALYSIS_PROMPT_EN

        # Singleton pattern: share model across instances to save VRAM
        if VLMAnalyzer._instance_model is None:
            print(f"[VLMAnalyzer] Loading model: {self.model_id}")
            print("  (First run will download ~4GB weights from HuggingFace)")

            # Determine target device explicitly (avoid meta tensor issue with device_map='auto')
            if device == "auto":
                target_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                target_device = device

            # Choose appropriate dtype (T4 supports float16 better than bfloat16)
            if target_device == "cuda":
                dtype = torch.float16
            else:
                dtype = torch.float32
            
            print(f"[VLMAnalyzer] Loading to {target_device} with dtype {dtype}")

            try:
                import os
                import sys
                
                # Prevent accelerate from using meta tensors
                os.environ.pop('ACCELERATE_USE_FAST_INIT', None)
                
                VLMAnalyzer._instance_tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                )
                
                # CRITICAL FIX 6: Add missing clean_up_tokenization method to InternLM2Tokenizer
                # This method is called during decode() but not implemented in InternLM2Tokenizer
                if not hasattr(VLMAnalyzer._instance_tokenizer, 'clean_up_tokenization'):
                    def clean_up_tokenization(self, out_string: str) -> str:
                        """Clean up tokenization artifacts after decoding."""
                        # Remove extra spaces before punctuation
                        out_string = out_string.replace(" .", ".").replace(" ?", "?").replace(" !", "!")
                        out_string = out_string.replace(" ,", ",").replace(" :", ":").replace(" ;", ";")
                        # Fix contractions
                        out_string = out_string.replace(" '", "'").replace(" n't", "n't")
                        out_string = out_string.replace(" 'm", "'m").replace(" 's", "'s")
                        out_string = out_string.replace(" 've", "'ve").replace(" 're", "'re")
                        return out_string.strip()
                    
                    # Bind the method to the tokenizer instance
                    import types
                    VLMAnalyzer._instance_tokenizer.clean_up_tokenization = types.MethodType(
                        clean_up_tokenization, 
                        VLMAnalyzer._instance_tokenizer
                    )
                    print(f"[VLMAnalyzer] Added clean_up_tokenization method to tokenizer")
                
                print(f"[VLMAnalyzer] Tokenizer loaded successfully")
                
                print(f"[VLMAnalyzer] Loading model weights (this may take 1-2 minutes)...")
                
                # CRITICAL FIX: Patch to prevent meta tensor issues in InternVL2's __init__
                # The model's code calls torch.linspace().item() which fails on meta tensors
                # We temporarily monkey-patch torch.linspace to ensure it never creates meta tensors
                original_linspace = torch.linspace
                def safe_linspace(*args, **kwargs):
                    # Force device to be non-meta
                    if 'device' not in kwargs or kwargs.get('device') == torch.device('meta'):
                        kwargs['device'] = 'cpu'
                    return original_linspace(*args, **kwargs)
                
                torch.linspace = safe_linspace
                
                # CRITICAL FIX 2: Patch transformers' tied weights check for InternVL2 compatibility
                # InternVL2 uses _tied_weights_keys but transformers>=4.50 expects all_tied_weights_keys
                from transformers import PreTrainedModel
                original_mark_tied = PreTrainedModel.mark_tied_weights_as_initialized
                
                def safe_mark_tied_weights(self):
                    try:
                        # Try to add missing attribute if needed
                        if not hasattr(self, 'all_tied_weights_keys'):
                            # Convert _tied_weights_keys to all_tied_weights_keys format
                            tied_keys = getattr(self, '_tied_weights_keys', None)
                            if tied_keys is not None and tied_keys:
                                self.all_tied_weights_keys = {k: None for k in tied_keys}
                            else:
                                # Empty dict if no tied weights
                                self.all_tied_weights_keys = {}
                        return original_mark_tied(self)
                    except (AttributeError, TypeError) as e:
                        # If still failing, just skip tied weights initialization
                        print(f"[VLMAnalyzer] Warning: Skipping tied weights initialization: {e}")
                        return
                
                PreTrainedModel.mark_tied_weights_as_initialized = safe_mark_tied_weights
                
                try:
                    # Load model WITHOUT device_map to avoid meta tensor initialization
                    # This uses the traditional loading path without accelerate's optimization
                    if target_device == "cuda":
                        # Load to GPU directly with target dtype
                        try:
                            VLMAnalyzer._instance_model = AutoModel.from_pretrained(
                                self.model_id,
                                dtype=dtype,
                                trust_remote_code=True,
                            ).to(target_device)
                        except TypeError:
                            # Backward compatibility for older transformers signatures
                            VLMAnalyzer._instance_model = AutoModel.from_pretrained(
                                self.model_id,
                                torch_dtype=dtype,
                                trust_remote_code=True,
                            ).to(target_device)
                    else:
                        # Load to CPU
                        try:
                            VLMAnalyzer._instance_model = AutoModel.from_pretrained(
                                self.model_id,
                                dtype=dtype,
                                trust_remote_code=True,
                            )
                        except TypeError:
                            VLMAnalyzer._instance_model = AutoModel.from_pretrained(
                                self.model_id,
                                torch_dtype=dtype,
                                trust_remote_code=True,
                            )
                    
                    # CRITICAL FIX 4: Make InternLM2ForCausalLM inherit from GenerationMixin
                    # transformers>=4.50 removed automatic GenerationMixin inheritance
                    # Solution: Dynamically add GenerationMixin to the class's base classes
                    from transformers.generation.utils import GenerationMixin
                    from transformers import GenerationConfig
                    
                    # The InternVL model has a language_model attribute (InternLM2ForCausalLM)
                    if hasattr(VLMAnalyzer._instance_model, 'language_model'):
                        lm = VLMAnalyzer._instance_model.language_model
                        lm_class = lm.__class__
                        
                        # Check if class already inherits from GenerationMixin
                        if not issubclass(lm_class, GenerationMixin):
                            # Modify the class hierarchy to include GenerationMixin
                            # This ensures all methods (instance, class, static, properties) are available
                            lm_class.__bases__ = (GenerationMixin,) + lm_class.__bases__
                            print(f"[VLMAnalyzer] Added GenerationMixin to {lm_class.__name__} class hierarchy")
                        
                        # Add generation_config if missing
                        if not hasattr(lm, 'generation_config'):
                            # Try to load from model or use default
                            try:
                                lm.generation_config = GenerationConfig.from_pretrained(self.model_id)
                            except:
                                # Use default generation config
                                lm.generation_config = GenerationConfig()
                            print(f"[VLMAnalyzer] Added generation_config to language_model")
                    
                finally:
                    # Restore original functions
                    torch.linspace = original_linspace
                    PreTrainedModel.mark_tied_weights_as_initialized = original_mark_tied
                
                VLMAnalyzer._instance_model.eval()
                
                # Save device and dtype info to avoid querying parameters later
                VLMAnalyzer._instance_device = target_device
                VLMAnalyzer._instance_dtype = dtype
                
                # Verify model is properly loaded
                try:
                    param_count = sum(p.numel() for p in VLMAnalyzer._instance_model.parameters() if not p.is_meta)
                    print(f"[VLMAnalyzer] Model loaded successfully with {param_count:,} parameters on {target_device}")
                except:
                    # If parameter counting fails, just report success
                    print(f"[VLMAnalyzer] Model loaded successfully on {target_device}")
                
            except Exception as e:
                print(f"[VLMAnalyzer] ERROR during model loading: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        self.model = VLMAnalyzer._instance_model
        self.tokenizer = VLMAnalyzer._instance_tokenizer
        self.device = VLMAnalyzer._instance_device
        self.dtype = VLMAnalyzer._instance_dtype

    def analyze_keyframe(self, image_path: str) -> dict:
        """
        Analyze a single keyframe image.
        
        Args:
            image_path: Path to the JPEG/PNG keyframe.
            
        Returns:
            dict: Structured analysis with keys: subject, action, visual_style,
                  lighting, camera_angle, color_palette, background, 
                  text_overlays, scaffolding_cues, overall_quality
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print(f"[VLMAnalyzer] Analyzing: {os.path.basename(image_path)}")
        
        # Load and preprocess image (use saved device/dtype info)
        pixel_values = _load_image(image_path).to(
            dtype=self.dtype,
            device=self.device,
        )

        # Generation config - InternVL2's chat() expects a dict.
        # Use anti-repetition settings to avoid degenerate loops like "清水清水...".
        generation_config = {
            "max_new_tokens": 320,
            "do_sample": True,
            "temperature": 0.2,
            "top_p": 0.9,
            "repetition_penalty": 1.12,
            "no_repeat_ngram_size": 4,
        }

        # 1) Primary attempt with full analysis prompt
        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            self.prompt,
            generation_config,
        )
        parsed = self._try_parse_response(response)
        if parsed is not None and not self._is_degenerate_response(response):
            parsed["_source_frame"] = image_path
            return parsed

        # 2) Retry with stronger structure constraint if first output is malformed/degenerate
        retry_prompt = RETRY_PROMPT_ZH if self.lang == "zh" else RETRY_PROMPT_EN
        retry_config = {
            "max_new_tokens": 220,
            "do_sample": False,
            "repetition_penalty": 1.15,
            "no_repeat_ngram_size": 4,
        }
        retry_response = self.model.chat(
            self.tokenizer,
            pixel_values,
            retry_prompt,
            retry_config,
        )
        retry_parsed = self._try_parse_response(retry_response)
        if retry_parsed is not None:
            retry_parsed["_source_frame"] = image_path
            return retry_parsed

        print("  [WARN] Could not parse JSON after retry, returning fallback object")
        return self._fallback_result(retry_response, image_path)

    def analyze_batch(self, image_paths: list[str]) -> list[dict]:
        """
        Analyze multiple keyframes sequentially.
        
        Args:
            image_paths: List of paths to keyframe images.
            
        Returns:
            List of analysis dicts.
        """
        results = []
        for i, path in enumerate(image_paths):
            print(f"[VLMAnalyzer] Frame {i+1}/{len(image_paths)}")
            try:
                result = self.analyze_keyframe(path)
                result["_source_frame"] = path
                results.append(result)
            except Exception as e:
                print(f"  [WARN] Failed to analyze {path}: {e}")
                results.append({"_source_frame": path, "_error": str(e)})
        return results

    def _sanitize_response(self, response: str) -> str:
        """Strip wrappers and keep only plausible JSON text."""
        clean = (response or "").strip()
        clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s*```$", "", clean)
        return clean.strip()

    def _extract_json_block(self, text: str) -> str | None:
        """Extract first balanced JSON object block from arbitrary text."""
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    def _normalize_result(self, data: dict) -> dict:
        """Normalize model output into the fixed schema expected downstream."""
        normalized = {}
        for key in self.EXPECTED_KEYS:
            value = data.get(key, "unknown")
            if value is None:
                value = "unknown"
            if not isinstance(value, str):
                value = str(value)
            normalized[key] = value.strip() if value.strip() else "unknown"
        return normalized

    def _try_parse_response(self, response: str) -> dict | None:
        """Best-effort parse of JSON response. Returns None when parsing fails."""
        clean = self._sanitize_response(response)

        # 1) Parse whole string
        try:
            obj = json.loads(clean)
            if isinstance(obj, dict):
                return self._normalize_result(obj)
        except json.JSONDecodeError:
            pass

        # 2) Parse extracted balanced JSON block
        block = self._extract_json_block(clean)
        if block:
            try:
                obj = json.loads(block)
                if isinstance(obj, dict):
                    return self._normalize_result(obj)
            except json.JSONDecodeError:
                return None
        return None

    def _is_degenerate_response(self, response: str) -> bool:
        """Detect obvious repetitive collapse in generated text."""
        text = (response or "").strip()
        if not text:
            return True

        # Repeated short token patterns, e.g. "清水" repeated many times
        if re.search(r"(.{1,4})\1{10,}", text):
            return True

        # Very low diversity is usually a collapsed generation
        short = text[:200]
        unique_chars = len(set(short))
        if len(short) >= 80 and unique_chars <= 8:
            return True
        return False

    def _fallback_result(self, response: str, image_path: str) -> dict:
        """Fallback object when response is still unparsable after retry."""
        result = {key: "unknown" for key in self.EXPECTED_KEYS}
        result["_raw_response"] = response
        result["_source_frame"] = image_path
        return result


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python vlm_analyzer.py <image_path.jpg>")
        print("Example: python vlm_analyzer.py data/keyframes/scene_001_keyframe.jpg")
        sys.exit(1)
    
    analyzer = VLMAnalyzer(lang="en")
    result = analyzer.analyze_keyframe(sys.argv[1])
    print("\n=== VLM Analysis Result ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
