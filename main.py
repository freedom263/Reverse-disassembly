import os
from src.utils.video_processor import process_video
from src.perception.vlm_analyzer import VLMAnalyzer
from src.agents.pedagogy_agent import PedagogyAgent
from src.agents.art_agent import ArtDirectorAgent
from src.agents.prompt_synthesizer import PromptSynthesizer

def main():
    print("=== Starting Upward Tracing Reverse System ===")
    
    # 0. Configuration
    data_dir = os.path.join("data")
    raw_video_dir = os.path.join(data_dir, "raw_videos")
    keyframe_dir = os.path.join(data_dir, "keyframes")
    
    # Create necessary directories if they don't exist
    for d in [raw_video_dir, keyframe_dir, os.path.join(data_dir, "golden_dataset")]:
        os.makedirs(d, exist_ok=True)
        
    print(f"Project directories verified at {os.path.abspath(data_dir)}")

    # Check for test video
    test_video_path = os.path.join(raw_video_dir, "test_video.mp4")
    if not os.path.exists(test_video_path):
        print(f"\n[WARNING] No test video found at {test_video_path}")
        print("Please place a sample video named 'test_video.mp4' in data/raw_videos/")
        print("Falling back to dry-run mode without real keyframes.\n")
        dummy_run = True
    else:
        dummy_run = False

    # 1. Perception Layer (Stage 1)
    print("\n--- Stage 1: Perception & Data Engineering ---")
    keyframes = []
    if not dummy_run:
        keyframes = process_video(test_video_path, keyframe_dir)
        print(f"Extracted {len(keyframes)} keyframes.")
    else:
        print("Skipping video processing (dummy run). Using mock keyframe.")
        keyframes = ["mock_keyframe.jpg"]

    vlm = VLMAnalyzer(use_mock=True) # Using mock for initial setup
    
    # Process the first keyframe for demonstration
    if keyframes:
        target_frame = keyframes[0]
        vlm_output = vlm.analyze_keyframe(target_frame)
        print("VLM Output:", vlm_output)

        # 2. Reasoning Layer (Stage 2)
        print("\n--- Stage 2: Reasoning Agents ---")
        
        pedagogy_agent = PedagogyAgent(use_mock=True)
        pedagogy_intent = pedagogy_agent.analyze(vlm_output)
        print("Pedagogy Intent:", pedagogy_intent)
        
        art_agent = ArtDirectorAgent(use_mock=True)
        art_style = art_agent.analyze(vlm_output)
        print("Art Style:", art_style)
        
        synthesizer = PromptSynthesizer()
        final_prompt = synthesizer.synthesize(pedagogy_intent, art_style, vlm_output)
        
        print("\n--- Final Synthesized Prompt ---")
        print(final_prompt)
        print("--------------------------------")

    # 3. Pattern Mining (Stage 3 - basic placeholder call)
    print("\n--- Stage 3: Pattern Mining Placeholder ---")
    print("Run `python -m src.analysis.pattern_miner` to test the mining module.")

    print("\n=== Initial Setup Complete ===")

if __name__ == "__main__":
    main()
