import os
import subprocess
import tempfile
from scenedetect import detect, ContentDetector
import cv2


def _transcode_to_h264(video_path: str) -> str | None:
    """
    Transcode a video to H264 using ffmpeg if OpenCV can't open it (e.g. AV1).
    Returns the path to the transcoded file, or None on failure.
    """
    out_path = video_path.rsplit(".", 1)[0] + "_h264.mp4"
    if os.path.exists(out_path):
        return out_path
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac",
        "-loglevel", "error",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=300)
        print(f"  [Transcode] AV1→H264: {os.path.basename(out_path)}")
        return out_path
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"  [Transcode] ffmpeg failed: {e}")
        return None


def _open_video(video_path: str):
    """
    Open a video with OpenCV. If the codec is unsupported (AV1 etc),
    automatically transcode to H264 first.
    Returns (cap, actual_path_used).
    """
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        # Try to actually read a frame - some codecs open but fail to decode
        ret, _ = cap.read()
        if ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return cap, video_path
        cap.release()

    print(f"  [VideoProcessor] Cannot decode {os.path.basename(video_path)}, trying ffmpeg transcode...")
    h264_path = _transcode_to_h264(video_path)
    if h264_path:
        cap2 = cv2.VideoCapture(h264_path)
        if cap2.isOpened():
            return cap2, h264_path

    print(f"  [VideoProcessor] Failed to open video: {video_path}")
    return None, video_path


def extract_scenes(video_path, threshold=27.0):
    """
    Extracts scenes from a video using PySceneDetect.
    Automatically handles AV1/unsupported codecs by transcoding first.

    Args:
        video_path (str): Path to the video file.
        threshold (float): Threshold for scene change detection.

    Returns:
        list of tuples: (start_time, end_time) of each scene.
        Also returns the actual video path used (may be transcoded copy).
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return [], video_path

    # Check if OpenCV can open it; transcode if needed
    cap, actual_path = _open_video(video_path)
    if cap is not None:
        cap.release()
    else:
        return [], video_path

    print(f"Analyzing video for scenes: {actual_path}")
    try:
        scene_list = detect(actual_path, ContentDetector(threshold=threshold))
    except Exception as e:
        print(f"  [VideoProcessor] Scene detect failed: {e}")
        return [], actual_path

    scenes = []
    for i, scene in enumerate(scene_list):
        print(f"  Scene {i+1}: {scene[0].get_timecode()} → {scene[1].get_timecode()}")
        scenes.append((scene[0], scene[1]))

    return scenes, actual_path


def save_keyframe(video_path, scene, output_dir, scene_index):
    """
    Saves the middle frame of a scene as a keyframe JPEG.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap, _ = _open_video(video_path)
    if cap is None:
        return None

    start_frame = scene[0].get_frames()
    end_frame = scene[1].get_frames()
    middle_frame = int((start_frame + end_frame) / 2)

    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    ret, frame = cap.read()
    cap.release()

    if ret:
        output_filename = os.path.join(output_dir, f"scene_{scene_index:03d}_keyframe.jpg")
        cv2.imwrite(output_filename, frame)
        print(f"Saved keyframe: {output_filename}")
        return output_filename
    else:
        print(f"Error: Could not read frame {middle_frame} from {video_path}")
        return None


def process_video(video_path, keyframe_dir):
    """
    Process a single video: detect scenes and save one keyframe per scene.
    Returns list of saved keyframe paths.
    """
    print(f"--- Processing: {video_path} ---")
    scenes, actual_path = extract_scenes(video_path)

    if not scenes:
        # Fallback: uniform sampling if scene detection yields nothing
        print(f"  [Fallback] Uniform frame sampling (every 30s)")
        return _uniform_sample(actual_path, keyframe_dir)

    saved_frames = []
    for i, scene in enumerate(scenes):
        frame_path = save_keyframe(actual_path, scene, keyframe_dir, i)
        if frame_path:
            saved_frames.append(frame_path)

    print(f"[Pipeline] Extracted {len(saved_frames)} keyframes")
    return saved_frames


def _uniform_sample(video_path: str, output_dir: str, interval_sec: int = 30) -> list:
    """
    Fallback: sample one frame every `interval_sec` seconds.
    Used when scene detection yields zero scenes.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = int(fps * interval_sec)
    saved = []
    idx = 0

    for frame_no in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if ret:
            out_path = os.path.join(output_dir, f"scene_{idx:03d}_keyframe.jpg")
            cv2.imwrite(out_path, frame)
            saved.append(out_path)
            idx += 1

    cap.release()
    print(f"  [Fallback] Sampled {len(saved)} frames")
    return saved


if __name__ == "__main__":
    pass
