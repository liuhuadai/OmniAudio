import os
import subprocess
from tqdm import tqdm


def filter_list(video_ids, filter_ids, block=False):
    """
    Filter video IDs from a list based on specified criteria
    
    :param video_ids: List containing video IDs to be filtered
    :param filter_ids: List of video IDs to use as filter
    :param block: If True, excludes specified video IDs. If False, includes only specified video IDs
    :return: Filtered list of video IDs
    """
    filter_ids = set(filter_ids)
    if block:
        return [vid for vid in video_ids if vid not in filter_ids]
    else:
        return [vid for vid in video_ids if vid in filter_ids]


def filter_size(file_dir, file_ids, ext, size_limit=1024, filter_less=True):
    """
    Filter files in a directory based on file size
    
    :param file_dir: Directory path containing files
    :param file_ids: List of file IDs to check
    :param ext: File extension (e.g. '.mp4', '.avi')
    :param size_limit: Size threshold in bytes
    :param filter_less: If True, keeps files >= size_limit. If False, keeps files < size_limit
    :return: List of file IDs that meet the size criteria
    """
    if not ext.endswith("."):
        ext = "." + ext
    output_file_ids = []
    for file_id in file_ids:
        file_path = os.path.join(file_dir, file_id + ext)
        if not os.path.exists(file_path):
            continue
        file_size = os.path.getsize(file_path)
        if filter_less and file_size >= size_limit:
            output_file_ids.append(file_id)
        elif not filter_less and file_size < size_limit:
            output_file_ids.append(file_id)
    return output_file_ids


def filter_audio_channels(file_dir, file_ids, ext, num_channels):
    """
    Filter files based on number of audio channels using ffmpeg
    
    :param file_dir: Directory containing files
    :param file_ids: List of file IDs to check
    :param ext: File extension (e.g. '.mp3', '.wav')
    :param num_channels: Target number of audio channels
    :return: List of file IDs matching the channel count criteria
    """
    if not ext.startswith("."):
        ext = "." + ext
        
    matching_files = []

    for file_id in tqdm(file_ids):
        # Construct full file path
        file_path = os.path.join(file_dir, f"{file_id}{ext}")

        # Verify file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        try:
            # Use ffprobe to check audio channel count
            cmd = [
                "ffprobe",
                "-v",
                "error",  # Suppress verbose output
                "-select_streams",
                "a:0",  # Select first audio stream
                "-show_entries",
                "stream=channels",  # Get channel count
                "-of",
                "csv=p=0",  # Format output
                file_path,
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True
            )

            # Get channel count
            channels = int(result.stdout.strip())
            if channels == num_channels:
                matching_files.append(file_id)
        except subprocess.CalledProcessError as e:
            print(f"Error processing file {file_path}: {e}")
        except ValueError as e:
            print(f"Invalid channel count for file {file_path}: {e}")

    return matching_files