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


class VLMAnalyzer:
    """
    Analyzes video keyframes using InternVL2-2B running locally on GPU.
    
    On first instantiation, the model (~4GB) is downloaded from HuggingFace.
    Subsequent runs load from the local HuggingFace cache.
    """

    MODEL_ID = "OpenGVLab/InternVL2-2B"
    _instance_model = None
    _instance_tokenizer = None

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

            VLMAnalyzer._instance_tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )
            
            # Load model with low_cpu_mem_usage=False to avoid meta tensors
            # This ensures full weight initialization before moving to device
            VLMAnalyzer._instance_model = AutoModel.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=False,  # Critical: avoid meta tensors
                trust_remote_code=True,
            ).to(target_device).eval()
            
            print(f"[VLMAnalyzer] Model loaded successfully on: {next(VLMAnalyzer._instance_model.parameters()).device}")
        
        self.model = VLMAnalyzer._instance_model
        self.tokenizer = VLMAnalyzer._instance_tokenizer

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
        
        model_device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        
        # Load and preprocess image
        pixel_values = _load_image(image_path).to(
            dtype=model_dtype,
            device=model_device,
        )
        
        # Generation config
        generation_config = {
            "max_new_tokens": 512,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
        }
        
        # Run inference
        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            self.prompt,
            generation_config,
        )
        
        # Parse JSON from response
        result = self._parse_response(response, image_path)
        return result

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

    def _parse_response(self, response: str, image_path: str) -> dict:
        """Extract JSON dict from model response string."""
        # Try to find JSON block in response
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Fallback: try parsing entire response
        try:
            clean = response.strip()
            if clean.startswith("```"):
                clean = re.sub(r'^```[a-z]*\n?', '', clean)
                clean = re.sub(r'\n?```$', '', clean)
            return json.loads(clean)
        except json.JSONDecodeError:
            print(f"  [WARN] Could not parse JSON, returning raw response")
            return {
                "subject": "unknown",
                "visual_style": "unknown",
                "overall_quality": "unknown",
                "_raw_response": response,
                "_source_frame": image_path,
            }


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
