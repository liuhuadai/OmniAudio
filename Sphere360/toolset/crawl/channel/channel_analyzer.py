import csv
import os
from collections import Counter
from itertools import islice
import sys
sys.path.append('..')
from core import build

youtube = build.build_youtube()

def batch(iterable, size):
    """
    Split an iterable into chunks of specified size.
    """
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk


def get_channel_info_batch(video_ids):
    """
    Batch retrieve channel IDs and names using video IDs.
    """
    channel_info = []
    try:
        response = (
            youtube.videos()
            .list(part="snippet", id=",".join(video_ids))
            .execute()
        )
        for item in response.get("items", []):
            snippet = item["snippet"]
            channel_info.append((snippet["channelId"], snippet["channelTitle"]))
    except Exception as e:
        print(f"Error fetching data for video_ids {video_ids}: {e}")
    return channel_info


def process_single_csv(input_file):
    """
    Process a single CSV file to:
    1. Extract video IDs
    2. Query YouTube API for channel info
    3. Count channel occurrences
    4. Return sorted results
    """
    video_ids = []
    with open(input_file, "r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            video_ids.append(row["video_id"])  # Assuming CSV has video_id column

    # Batch process channel info
    channel_counter = Counter()
    channel_details = {}

    for video_batch in batch(video_ids, 50):  # Process 50 video IDs per batch
        channel_info_batch = get_channel_info_batch(video_batch)
        for channel_id, channel_title in channel_info_batch:
            channel_counter[channel_id] += 1
            channel_details[channel_id] = channel_title

    # Sort by occurrence count
    sorted_channels = channel_counter.most_common()

    return sorted_channels, channel_details


def process_folder(folder_path, output_file):
    """
    Process all CSV files in a folder to:
    1. Aggregate video IDs
    2. Collect channel statistics
    3. Merge results
    4. Output sorted results to CSV
    """
    all_channel_counter = Counter()
    all_channel_details = {}

    # Process each CSV file in folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            input_file = os.path.join(folder_path, filename)
            print(f"Processing file: {input_file}")

            # Process individual CSV
            sorted_channels, channel_details = process_single_csv(input_file)

            # Aggregate results
            for channel_id, count in sorted_channels:
                all_channel_counter[channel_id] += count
            all_channel_details.update(channel_details)

    # Write final results to CSV
    with open(output_file, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.writer(outfile, quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writerow(["channel_id", "channel_name", "count"])
        for channel_id, count in all_channel_counter.most_common():
            writer.writerow(
                [channel_id, all_channel_details[channel_id], count]
            )


# Example execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-dir",
        help="Path to folder containing CSV files",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-csv",
        help="Output CSV file path",
        default="output.csv",
    )
    args = parser.parse_args()
    folder_path = args.input_dir  # CSV files directory
    output_csv = args.output_csv  # Output file path
    process_folder(folder_path, output_csv)