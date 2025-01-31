

import os
import cv2

def extract_frames(video_path, frame_rate, output_path="ponedijak_popodne"):
    """
    Extract frames from a local video file at the given frame rate.

    Args:
        video_path (str): Path to the video file.
        frame_rate (int): Number of frames per second to extract.
        output_path (str): Directory where frames will be saved.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load the video
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get the video's frame rate
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frame_rate)

    # Initialize counters
    success, frame = video_capture.read()
    frame_count = 0
    saved_frame_count = 0

    while success:
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_path, f"frame_{saved_frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
        success, frame = video_capture.read()
        frame_count += 1

    video_capture.release()
    print(f"Extracted {saved_frame_count} frames to {output_path}")

# Example usage
if __name__ == "__main__":
    video_file = "C:/Users/Korisnik/Downloads/100 People Give Us Their Hot Take ｜ Keep it 100 ｜ Cut.mp4"
  # Path to the local video file
    frames_per_second = 1  # Number of frames to extract per second
    extract_frames(video_file, frames_per_second)

