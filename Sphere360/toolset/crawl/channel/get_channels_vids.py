"""
Collect all 360-degree video IDs from specified channels and output to CSV
"""
import os
import sys
import csv
import argparse
from tqdm import tqdm

sys.path.append("..")
import core
from core import build, channel

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Collect 360-degree videos from YouTube channels"
    )
    parser.add_argument(
        "-i", "--input-csv",
        type=str,
        required=True,
        help="Input CSV file containing channel IDs"
    )
    parser.add_argument(
        "-o", "--output-csv",
        type=str,
        required=True,
        help="Output CSV file for 360-degree video IDs"
    )
    args = parser.parse_args()

    # Initialize YouTube API client
    youtube = build.build_youtube()

    # Get channel IDs from input file
    channel_ids = core.filelist.get_channel_ids(args.input_csv)

    # Collect all 360-degree video IDs
    video_ids = []
    for channel_id in tqdm(channel_ids, desc="Processing channels"):
        cur_video_ids = channel.get_channel_video_ids_360(youtube, channel_id)
        print(f"Channel {channel_id}: Found {len(cur_video_ids)} 360-degree videos")
        video_ids.extend(cur_video_ids)
    
    # Write results to CSV
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id"])
        writer.writerows([[vid] for vid in video_ids])
    
    print(f"\nCompleted. Saved {len(video_ids)} 360-degree video IDs to {args.output_csv}")