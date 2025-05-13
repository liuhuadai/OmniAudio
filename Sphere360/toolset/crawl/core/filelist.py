import csv
import os

def get_channel_ids(input_file: str):
    channel_ids = []
    if input_file.endswith(".csv"):  # CSV file
        with open(input_file, "r") as file:
            csvreader = csv.DictReader(file)
            for row in csvreader:
                channel_ids.append(row["channel_id"])
    else:  # Default format
        with open(input_file, "r") as file:
            lines = file.readlines()
            for line in lines:
                channel_ids.append(line.strip())

    return channel_ids

def get_video_ids(input_file: str):
    video_ids = []
    if input_file.endswith(".csv"):  # CSV file
        with open(input_file, "r") as file:
            csvreader = csv.DictReader(file)
            for row in csvreader:
                video_ids.append(row["video_id"])
    else:  # Default format
        with open(input_file, "r") as file:
            lines = file.readlines()
            for line in lines:
                video_ids.append(line.strip())

    return video_ids

def get_file_ids(input_file: str):
    file_ids = []
    if input_file.endswith(".csv"):  # CSV file
        with open(input_file, "r") as file:
            csvreader = csv.reader(file)
            # skip header
            next(csvreader)
            for row in csvreader:
                file_ids.append(row[0])
    else:  # Default format
        with open(input_file, "r") as file:
            lines = file.readlines()
            for line in lines:
                file_ids.append(line.strip())

    return file_ids

def get_video_ids_and_start_times(input_file: str):
    video_ids = []
    start_times = []
    lines = []
    if input_file.endswith(".csv"):  # CSV file
        with open(input_file, "r") as file:
            csvreader = csv.DictReader(file)
            for row in csvreader:
                lines.append(row["file_id"])
    else:  # Default format
        with open(input_file, "r") as file:
            lines = file.readlines()

    for line in lines:
        parts = line.rsplit("_", 1)
        if len(parts) != 2:
            continue
        video_ids.append(parts[0])
        start_times.append(int(parts[1]))

    return video_ids, start_times

def get_video_ids_and_start_times_list(input_file: str):
    video_data = {}  # Dictionary to store video_id and corresponding start_times_list
    
    with open(input_file, "r") as file:
        lines = file.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 2:
            continue

        video_id = parts[0]
        start_times = parts[1].split(",")
        start_times_list = [int(t) for t in start_times if t.isdigit()]
        
        # Merge start_times_list for same video_id
        if video_id in video_data:
            video_data[video_id].extend(start_times_list)
        else:
            video_data[video_id] = start_times_list
    
    # Deduplicate and sort start_times_list for each video_id
    for video_id in video_data:
        video_data[video_id] = sorted(set(video_data[video_id]))
    
    # Separate results into two lists
    file_ids = list(video_data.keys())
    start_times_list = list(video_data.values())
    
    return file_ids, start_times_list

def get_video_ids_from_dir(input_dir: str, ext: str):
    """
    Extract video IDs from filenames in the specified directory.
    
    Args:
        input_dir: Directory containing the files
        ext: File extension to filter by (e.g., 'mp4')
        
    Returns:
        List of video IDs extracted from filenames
    """
    if not ext.startswith("."):
        ext = "." + ext
    video_ids = []
    for file in os.listdir(input_dir):
        if file.endswith(ext):
            video_ids.append(file.split(".")[0])
    return video_ids

def get_video_ids_from_dir_and_start_times(input_dir: str, ext: str):
    """
    Extract video IDs and start times from filenames in the specified directory.
    
    Args:
        input_dir: Directory containing the files
        ext: File extension to filter by (e.g., 'mp4')
        
    Returns:
        Tuple of (video_ids, start_times) extracted from filenames
    """
    if not ext.startswith("."):
        ext = "." + ext
    video_ids = []
    start_times = []
    for file in os.listdir(input_dir):
        if file.endswith(ext):
            parts = file.split(".")[0].rsplit("_", 1)
            if len(parts) != 2:
                continue
            video_ids.append(parts[0])
            start_times.append(int(parts[1]))
    return video_ids, start_times

def get_file_ids_from_dir(input_dir: str, ext: str):
    """
    Extract file IDs from filenames in the specified directory.
    
    Args:
        input_dir: Directory containing the files
        ext: File extension to filter by (e.g., 'mp4')
        
    Returns:
        List of file IDs extracted from filenames
    """
    if not ext.startswith("."):
        ext = "." + ext
    file_ids = []
    for file in os.listdir(input_dir):
        if file.endswith(ext):
            file_ids.append(file.split(".")[0])
    return file_ids