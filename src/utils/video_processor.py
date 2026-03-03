import os
from scenedetect import detect, ContentDetector
import cv2

def extract_scenes(video_path, threshold=27.0):
    """
    Extracts scenes from a video using PySceneDetect.
    
    Args:
        video_path (str): Path to the video file.
        threshold (float): Threshold for scene change detection.
        
    Returns:
        list of tuples: Each tuple contains (start_time, end_time) of a scene.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return []
        
    print(f"Analyzing video for scenes: {video_path}")
    scene_list = detect(video_path, ContentDetector(threshold=threshold))
    
    scenes = []
    for i, scene in enumerate(scene_list):
        print(f"  Scene {i+1}: Start {scene[0].get_timecode()} / Frame {scene[0].get_frames()} | "
              f"End {scene[1].get_timecode()} / Frame {scene[1].get_frames()}")
        scenes.append((scene[0], scene[1]))
        
    return scenes

def save_keyframe(video_path, scene, output_dir, scene_index):
    """
    Saves the middle frame of a scene as a keyframe.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
        
    start_frame = scene[0].get_frames()
    end_frame = scene[1].get_frames()
    middle_frame = int((start_frame + end_frame) / 2)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    ret, frame = cap.read()
    
    if ret:
        output_filename = os.path.join(output_dir, f"scene_{scene_index:03d}_keyframe.jpg")
        cv2.imwrite(output_filename, frame)
        print(f"Saved keyframe: {output_filename}")
        cap.release()
        return output_filename
    else:
        print(f"Error: Could not read frame {middle_frame}")
        cap.release()
        return None

def process_video(video_path, keyframe_dir):
    """
    Process a single video: extract scenes and save a keyframe for each.
    """
    print(f"--- Processing: {video_path} ---")
    scenes = extract_scenes(video_path)
    
    saved_frames = []
    for i, scene in enumerate(scenes):
        frame_path = save_keyframe(video_path, scene, keyframe_dir, i)
        if frame_path:
            saved_frames.append(frame_path)
            
    return saved_frames

if __name__ == "__main__":
    # Test script usage
    pass
