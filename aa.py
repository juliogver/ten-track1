from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
import cv2
import os

def debug_main():
    # Read Video
    input_video_path = "input_videos/RG - Trim.mp4"
    video_frames = read_video(input_video_path)
    
    video_filename = os.path.splitext(os.path.basename(input_video_path))[0]

    player_tracker = PlayerTracker(model_path='yolov8x')
    ball_tracker = BallTracker(model_path='models/last.pt')

    # Detect Players and Ball
    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=True)
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=True)

    # Draw output for debugging
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    # Save the debug video
    debug_output_path = "output_videos/debug_output_video.avi"
    save_video(output_video_frames, debug_output_path)

    print(f"Debug video saved at {debug_output_path}")

if __name__ == "__main__":
    debug_main()
