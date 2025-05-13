import cv2
import numpy as np
import os
from tqdm import tqdm
import re
import concurrent.futures

def get_video_duration(stderr_output):
    """Get the duration of a video using ffmpeg."""
    ffmpeg_output = stderr_output
    match = re.search(r"Duration: (\d{2}):(\d{2}):(\d{2})\.(\d{2})", ffmpeg_output, re.IGNORECASE)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        milliseconds = int(match.group(4))
        total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 100
        return total_seconds
    else:
        print("Duration not found in ffmpeg output.")
        return 0

def get_video_dimensions(stderr_output):
    """Extract video width and height from ffmpeg stderr output."""
    match = re.search(r'(\d{3,4})x(\d{3,4})', stderr_output)
    if match:
        width = int(match.group(1))
        height = int(match.group(2))
        return width, height
    return None, None

def get_video_fps(stderr_output):
    """Extract video frame rate from ffmpeg stderr output."""
    match = re.search(r'(\d+(\.\d+)?) fps', stderr_output)
    if match:
        fps = float(match.group(1))
        return fps
    return None

def calculate_mse(frame1, frame2):
    """Calculate Mean Squared Error between two frames."""
    frame1 = cv2.resize(frame1, (frame2.shape[1], frame2.shape[0]))  # Ensure consistent dimensions
    mse = np.sum((frame1 - frame2) ** 2) / float(frame1.size)  # Compute MSE
    return mse

def is_static(mse, threshold=5):  # Using slightly lower threshold
    """Determine if frame is static based on MSE threshold."""
    return mse < threshold  # Frame considered static if MSE below threshold

def load_frame(frame_path):
    """Load a frame and convert to grayscale."""
    frame = cv2.imread(frame_path)
    if frame is None:
        return None  # Return None if frame loading fails
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def detect_static_video(frame_path, frame_count=20):
    """Detect if video is static by analyzing frame differences."""
    try:
        static_frame_count = 0
        sample_count = 0

        # Generate frame file paths
        frame_paths = [os.path.join(frame_path, f'{frame_index*4:06d}.jpg') 
                      for frame_index in range(1, frame_count + 1)]

        # Parallel frame loading using multithreading
        with concurrent.futures.ThreadPoolExecutor() as executor:
            frames = list(executor.map(load_frame, frame_paths))

        # Filter out failed frame loads
        frames = [frame for frame in frames if frame is not None]

        # Require minimum 2 frames for analysis
        if len(frames) < 2:
            return False  # Insufficient frames for static detection

        # Count static frames
        prev_frame = frames[0]
        for gray_frame in frames[1:]:
            mse = calculate_mse(prev_frame, gray_frame)
            if is_static(mse):
                static_frame_count += 1
            prev_frame = gray_frame  # Update previous frame

        # Calculate static frame ratio
        sample_count = len(frames)  # Actual processed frame count
        static_ratio = static_frame_count / sample_count if sample_count > 0 else 0

        # Video considered static if >85% frames are static
        return static_ratio > 0.85

    except Exception as e:
        print(f"Error processing video: {frame_path}, Error: {e}")
        return None  # Return None to indicate error

def process_video_folder(folder_path, output_file, error_log):
    """Batch process video folders for static detection."""
    video_files = [f for f in os.listdir(folder_path) 
                  if re.search(r'^[a-zA-Z0-9_-]*_.\d*0$', f)]
    
    with open(output_file, 'w') as output:
        for idx, video_file in enumerate(tqdm(video_files, desc="Processing Videos", unit="file")):
            frame_path = os.path.join(folder_path, video_file)
            result = detect_static_video(frame_path)

            if result is None:
                # Log failed processing attempts
                with open(error_log, 'a') as error_output:
                    error_output.write(f"Error processing: {video_file}\n")
            elif result:
                output.write(f"{video_file}\n")  # Record static video filename
                print(f"{video_file} is static")

# Example usage
if __name__ == "__main__":
    folder_path = ''  # Video frame directory
    output_file = './static_new.txt'  # Static video output list
    error_log = './static_new_err.txt'  # Error log file

    process_video_folder(folder_path, output_file, error_log)