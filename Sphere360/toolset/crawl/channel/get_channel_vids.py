"""
Retrieve all 360-degree video IDs from channels with count >= threshold,
and output to corresponding directory.
"""
import os
import sys
import csv
from tqdm import tqdm
sys.path.append("..")
from core import build, channel

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Extract 360-degree videos from qualified channels"
    )
    parser.add_argument(
        "-i", "--input-csv",
        type=str,
        required=True,
        help="Input CSV file containing channel statistics"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        required=True,
        help="Output directory for video ID files"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=int,
        default=3,
        help="Minimum count threshold for channel inclusion"
    )
    parser.add_argument(
        "-d", "--database",
        type=str,
        default=None,
        help="Optional database CSV for channel filtering"
    )
    args = parser.parse_args()

    # Initialize YouTube API client
    youtube = build.build_youtube()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load database channel IDs if provided
    database = set()
    if args.database:
        with open(args.database, "r") as f:
            reader = csv.DictReader(f)
            database = {row["channel_id"] for row in reader}
    
    # Filter channels by threshold and database
    qualified_channels = []
    with open(args.input_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["count"]) >= args.threshold:
                if row["channel_id"] not in database:
                    qualified_channels.append(row["channel_id"])
                else:
                    print(f"Skipping {row['channel_name']}[{row['channel_id']}]")
    
    # Process qualified channels
    for channel_id in tqdm(qualified_channels, desc="Processing channels"):
        output_file = os.path.join(args.output_dir, f"{channel_id}.csv")
        
        # Skip if already processed
        if os.path.exists(output_file):
            print(f"Skipping existing: {channel_id}")
            continue
            
        # Get 360-degree video IDs
        video_ids = channel.get_channel_video_ids_360(youtube, channel_id)
        
        # Write to CSV
        with open(output_file, "w") as f:
            f.write("video_id\n")
            f.writelines(f"{vid}\n" for vid in video_ids)