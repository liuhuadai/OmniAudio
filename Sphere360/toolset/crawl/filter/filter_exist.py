import os
import csv
from tqdm import tqdm

# black-list database paths
# items within the video database or channel database will be filtered out
video_db_path = ''  # Path to video_id database file
channel_db_path = ''  # Path to channel_id database file

# Path to folder containing search results for filtering
folder_path = ''  # Replace with actual folder path
output_csv = ''  # Replace with actual output filename

def read_db(file_path, key_column):
    """Read values from a database CSV file"""
    values = set()
    with open(file_path, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.add(row[key_column])  # Get values from specified column
    return values

def process_csv_files(folder_path, video_db, channel_db):
    """Process all CSV files in the folder and filter results"""
    result = set()  # Stores final video_id results
    before_filter = set()  # Stores video_id results before filtering
    for file_name in tqdm(os.listdir(folder_path)):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, mode='r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    video_id = row['video_id']
                    channel_id = row['channel_id']
                    before_filter.add(video_id)
                    # Check if video_id and channel_id exist in databases
                    if video_id not in video_db and channel_id not in channel_db:
                        result.add(video_id)  # Add to results if not in databases
    return result, before_filter

def main():
    # Read databases
    video_db = read_db(video_db_path, 'video_id')
    channel_db = read_db(channel_db_path, 'channel_id')

    print(f"Video DB: {video_db_path} ({len(video_db)} records)")
    print(f"Channel DB: {channel_db_path} ({len(channel_db)} records)")

    # Process CSV files in folder
    result, before_filter = process_csv_files(folder_path, video_db, channel_db)

    # Output final results
    print(f"Number of video_id: {len(result)}/{len(before_filter)}")
    
    # Write pre-filter results
    with open(f"before_filter_{output_csv}", mode='w', newline='', encoding='utf-8') as f:
        dict_writer = csv.DictWriter(f, fieldnames=['video_id'])
        dict_writer.writeheader()
        for video_id in before_filter:
            dict_writer.writerow({'video_id': video_id})
    print(f"Before filter results saved to: before_filter_{output_csv}")

    # Write final filtered results
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        dict_writer = csv.DictWriter(f, fieldnames=['video_id'])
        dict_writer.writeheader()
        for video_id in result:
            dict_writer.writerow({'video_id': video_id})
    print(f"Filtered results saved to: {output_csv}")

if __name__ == "__main__":
    main()