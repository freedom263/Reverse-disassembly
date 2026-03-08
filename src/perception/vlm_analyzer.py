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

ANALYSIS_PROMPT_EN = """Analyze this image. Output ONLY this exact JSON:
{"subject":"main subject/object","action":"what is happening","visual_style":"art style like photorealistic, 3D animation, or cinematic","lighting":"lighting like soft studio or natural daylight","camera_angle":"camera angle like close-up or wide shot","color_palette":"dominant colors","background":"background type","text_overlays":"visible text or labels","scaffolding_cues":"pedagogical aids like arrows or highlights","overall_quality":"low, medium, high, or cinematic"}
Fill EVERY field. Use simple phrases. If uncertain, write "unknown". Output NOTHING except the JSON object."""

ANALYSIS_PROMPT_ZH = """分析这张图。仅输出这个确切的 JSON（不要其他任何文字）：
{"subject":"主体/对象","action":"正在发生的事","visual_style":"艺术风格如写实、3D动画、电影级","lighting":"光照如柔和或自然日光","camera_angle":"摄像角度如特写或广角","color_palette":"主要颜色","background":"背景类型","text_overlays":"可见的文字","scaffolding_cues":"教学辅助如箭头","overall_quality":"low、medium、high 或 cinematic"}
每个字段都要填。不确定时写 "unknown"。只输出 JSON，无其他文字。"""

RETRY_PROMPT_EN = """Output exactly this JSON (NOTHING else):
{"subject":"","action":"","visual_style":"","lighting":"","camera_angle":"","color_palette":"","background":"","text_overlays":"","scaffolding_cues":"","overall_quality":""}
For each field, write a short phrase or "unknown" if unsure."""

RETRY_PROMPT_ZH = """输出这个 JSON（只有这个）：
{"subject":"","action":"","visual_style":"","lighting":"","camera_angle":"","color_palette":"","background":"","text_overlays":"","scaffolding_cues":"","overall_quality":""}
每个字段填简短的词或 "unknown"。"""


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
                    
                    # REMOVED: GenerationMixin dynamic inheritance manipulation
                    # This was causing token generation to collapse into random characters.
                    # InternVL2's chat() method should work without this modification.
                    # If we need generate() functionality, we'll access it through the model's
                    # existing implementation rather than monkey-patching the class hierarchy.
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

        # ========== ATTEMPT 1: Call chat() with minimal, safe parameters ==========
        # Remove generation_config to let InternVL2 use its defaults
        try:
            response_1 = self.model.chat(
                self.tokenizer,
                pixel_values,
                self.prompt,
            )
            
            parsed_1 = self._try_parse_response(response_1)
            if parsed_1 is not None and not self._is_degenerate_response(response_1):
                parsed_1["_source_frame"] = image_path
                return parsed_1
        except Exception as e:
            print(f"  [WARN] chat() attempt 1 failed: {e}")
            response_1 = ""

        # ========== ATTEMPT 2: Try with explicit simple generation config ==========
        try:
            gen_config_minimal = {
                "max_new_tokens": 200,
            }
            response_2 = self.model.chat(
                self.tokenizer,
                pixel_values,
                self.prompt,
                gen_config_minimal,
            )
            
            parsed_2 = self._try_parse_response(response_2)
            if parsed_2 is not None:
                parsed_2["_source_frame"] = image_path
                return parsed_2
        except Exception as e:
            print(f"  [WARN] chat() attempt 2 failed: {e}")
            response_2 = ""

        # ========== ATTEMPT 3: Field-level extraction from raw text ==========
        print(f"  [DEBUG] Response 1 preview: {response_1[:150] if response_1 else 'EMPTY'}")
        print(f"  [DEBUG] Response 2 preview: {response_2[:150] if response_2 else 'EMPTY'}")
        
        field_extracted = self._extract_fields_from_text(response_2 or response_1)
        if field_extracted and any(v != "unknown" for v in field_extracted.values()):
            print(f"  [ATTEMPT3] Extracted {sum(1 for v in field_extracted.values() if v != 'unknown')} fields via regex")
            field_extracted["_source_frame"] = image_path
            field_extracted["_extraction_method"] = "field-level regex"
            return field_extracted

        print("  [WARN] All attempts failed, returning fallback object")
        return self._fallback_result(response_2 or response_1, image_path)

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

    def _extract_fields_from_text(self, text: str) -> dict | None:
        """
        Attempt to extract field values from arbitrary text using regex.
        Last-resort fallback when JSON parsing completely fails.
        """
        if not text or len(text.strip()) < 10:
            return None
        
        result = {key: "unknown" for key in self.EXPECTED_KEYS}
        
        # Try to find key-value patterns: "key: value" or "\"key\": \"value\""
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('[') or line.startswith('-'):
                continue
            
            # Match "key: value" patterns
            for key in self.EXPECTED_KEYS:
                # Pattern 1: "key: value"
                match = re.search(rf'{key}\s*[:\s]+([^,\n{{}}]+)', line, re.IGNORECASE)
                if match:
                    value = match.group(1).strip().strip('\'"')
                    if value and len(value) < 100:
                        result[key] = value
                        break
        
        # If we found at least 3 non-"unknown" fields, return it
        filled = sum(1 for v in result.values() if v != "unknown")
        if filled >= 3:
            return result
        
        return None


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
