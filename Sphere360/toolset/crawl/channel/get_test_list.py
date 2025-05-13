import os
import pandas as pd
import random
import argparse

def process_csv(input_folder, output_folder, sample_size=10):
    """
    Process CSV files in the input folder and randomly select a specified number of video_ids.
    
    Args:
        input_folder: Path to folder containing input CSV files
        output_folder: Path to folder where processed files will be saved
        sample_size: Number of video_ids to randomly select (default: 10)
    """
    # Get all CSV files in input folder
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    # Process each CSV file
    for csv_file in csv_files:
        # Build input and output file paths
        input_file_path = os.path.join(input_folder, csv_file)
        output_file_path = os.path.join(output_folder, csv_file)

        # Read CSV file
        df = pd.read_csv(input_file_path)

        # Check if 'video_id' column exists
        if 'video_id' not in df.columns:
            print(f"Warning: 'video_id' column not found in {csv_file}. Skipping...")
            continue

        # Randomly select specified number of video_ids
        all_video_ids = df['video_id'].tolist()
        if len(all_video_ids) > sample_size:
            selected_video_ids = random.sample(all_video_ids, sample_size)
        else:
            selected_video_ids = all_video_ids
            
        selected_file_ids = [video_id + '_5' for video_id in selected_video_ids]

        # Create new DataFrame with selected file_ids
        selected_df = pd.DataFrame({'file_id': selected_file_ids})

        # Save results to output folder
        selected_df.to_csv(output_file_path, index=False)

        print(f"Processed {csv_file}, selected {len(selected_video_ids)} video_ids, saved to {output_file_path}")

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Process CSV files and randomly select video_ids."
    )
    
    # Add command line arguments
    parser.add_argument('-i', '--input-folder', 
                       type=str, 
                       help="Input folder containing CSV files", 
                       required=True)
    parser.add_argument('-o', '--output-folder', 
                       type=str, 
                       help="Output folder to save the result CSV files", 
                       required=True)
    parser.add_argument('-n', '--sample-size',
                       type=int,
                       default=10,
                       help="Number of video_ids to randomly select (default: 10)")

    # Parse arguments
    args = parser.parse_args()

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Process CSV files
    process_csv(args.input_folder, args.output_folder, args.sample_size)

if __name__ == '__main__':
    main()