import json
import os

class VLMAnalyzer:
    def __init__(self, use_mock=True):
        """
        Initializes the VLM Analyzer.
        
        Args:
            use_mock (bool): If True, returns mock data instead of calling physical APIs (useful for dev/test without API keys or GPUs).
        """
        self.use_mock = use_mock
        # Initialize API clients here (e.g., openai.OpenAI()) if not mocking

    def analyze_keyframe(self, image_path):
        """
        Analyzes a keyframe to extract structured data.
        
        Args:
            image_path (str): Path to the keyframe image.
            
        Returns:
            dict: Structured JSON output containing visual elements, text content, and scaffolding cues.
        """
        print(f"Analyzing keyframe with VLM: {image_path}")
        
        if self.use_mock:
            # Mock analysis logic based on the schema mapping in the manual
            filename = os.path.basename(image_path)
            
            # Very basic mock response
            mock_response = {
                "visual_elements": ["Red arrow on map", "Zoom-in transition"],
                "text_content": [f"Title from {filename}"],
                "scaffolding_cues": ["Highlighting key terms", "Schematic diagram"]
            }
            return mock_response
            
        # REAL IMPLEMENTATION GOES HERE
        # e.g., using OpenAI client to analyze image with vision models
        # prompt = "你是一位教育技术专家。请分析这张教学视频截图。列出画面中所有的‘视觉支架’（如箭头、高亮）、教学内容的呈现方式（实拍/动画），以及当前的镜头语言。"
        
        raise NotImplementedError("Real VLM API call is not yet implemented.")

if __name__ == "__main__":
    # Test
    analyzer = VLMAnalyzer(use_mock=True)
    res = analyzer.analyze_keyframe("dummy_path.jpg")
    print(json.dumps(res, indent=2, ensure_ascii=False))
