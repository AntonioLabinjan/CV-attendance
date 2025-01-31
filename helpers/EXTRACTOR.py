import os
import cv2
from yt_dlp import YoutubeDL

def download_youtube_video(url, output_dir="downloads"):
    ydl_opts = {
        'format': 'best',  # Try the best available format
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),  # Save the video with its title and extension
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            video_path = os.path.join(output_dir, f"{info_dict['title']}.{info_dict['ext']}")
            print(f"Downloaded video: {video_path}")
            return video_path
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

def extract_frames(video_path, frame_rate, output_path="judi_od_nedije"):
    """Extract frames from a video at the given frame rate."""
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)  # Frames per second
    frame_interval = int(fps / frame_rate)  # Interval between frames to capture

    success, frame = video_capture.read()
    count = 0
    frame_count = 0

    while success:
        if count % frame_interval == 0:
            # Pad the frame number with leading zeros
            frame_filename = os.path.join(output_path, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1
        success, frame = video_capture.read()
        count += 1

    video_capture.release()
    print(f"Extracted {frame_count} frames to {output_path}")

if __name__ == "__main__":
    yt_link = "https://www.youtube.com/watch?v=ECoaaXtiPvI"  # Hardcoded YouTube link
    frame_rate = 1  # Frames per second
    output_dir = "downloads"
    
    # Download video from YouTube
    video_path = download_youtube_video(yt_link, output_dir)
    
    if video_path:
        # Extract frames if video download was successful
        extract_frames(video_path, frame_rate)
