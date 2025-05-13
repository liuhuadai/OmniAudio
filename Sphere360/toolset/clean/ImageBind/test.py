import torch
import os
import csv
from tqdm import tqdm
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from pathlib import Path
import pandas as pd
import random
import torch.nn.functional as F

# Set device: Use GPU if available, otherwise CPU
device = ""

# Load ImageBind model
try:
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)
except Exception as e:
    print(f"Error loading the model: {e}")
    exit(1)  # Exit if model loading fails

# Set audio and video folder paths
audio_folder = ""
video_folder = ""

# Read CSV file
csv_file = ''
try:
    df = pd.read_csv(csv_file)
except Exception as e:
    print(f"Error reading CSV file {csv_file}: {e}")
    exit(1)

# Prepare output CSV and error log files
output_csv = 'output.csv'
error_log_file = 'test.log'

# Track processed files by reading existing output (if any)
processed_files = set()
if os.path.exists(output_csv):
    try:
        with open(output_csv, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                file_name = row[0]  # Full filename
                processed_files.add(file_name)  # Record processed files
    except Exception as e:
        print(f"Error reading the output CSV file {output_csv}: {e}")
        exit(1)

# Initialize lists for matched audio-video pairs
paired_audio_paths = []
paired_video_paths = []

# Open error log for writing
with open(error_log_file, mode='a') as error_log:
    # Process each file_id, skipping already processed files
    for file_id in df['file_id']:
        audio_file = f"{file_id}.flac"
        video_file = f"000040.jpg"
        
        video_path = os.path.join(video_folder, file_id, video_file)
        audio_path = os.path.join(audio_folder, audio_file)

        # Get basenames without extensions
        video_name = os.path.basename(video_path)
        audio_name = os.path.basename(audio_path)
        video_name_no_ext = os.path.splitext(video_name)[0]
        audio_name_no_ext = os.path.splitext(audio_name)[0]

        # Skip if already processed
        if video_name_no_ext in processed_files or audio_name_no_ext in processed_files:
            continue

        # Validate file existence
        if not os.path.exists(video_path):
            error_log.write(f"Video directory not found: {video_path}\n")
            continue

        if not os.path.exists(audio_path):
            error_log.write(f"Audio file not found: {audio_path}\n")
            continue

        paired_audio_paths.append(audio_path)
        paired_video_paths.append(video_path)

print(f"Successfully matched {len(paired_audio_paths)} audio-video pairs.")

# Batch processing configuration
batch_size = 16 
num_batches = len(paired_video_paths) // batch_size + 1

# Process and write results
try:
    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)

        with torch.no_grad():
            for i in tqdm(range(num_batches), desc="Processing Batches"):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(paired_video_paths))

                # Get current batch paths
                video_batch_paths = paired_video_paths[start_idx:end_idx]
                audio_batch_paths = paired_audio_paths[start_idx:end_idx]

                try:
                    # Load batch data
                    video_batch = data.load_and_transform_vision_data(video_batch_paths, device) 
                    audio_batch = data.load_and_transform_audio_data(audio_batch_paths, device)
                except RuntimeError as e:
                    print(f"Error loading video data in batch {i}: {e}")
                    continue
                except Exception as e:
                    print(f"Unexpected error in batch {i}: {e}")
                    continue

                try:
                    # Model inference
                    inputs = {
                        ModalityType.VISION: video_batch,
                        ModalityType.AUDIO: audio_batch,
                    }

                    embeddings = model(inputs)

                    # Calculate similarity
                    audio_embedding = embeddings[ModalityType.AUDIO]
                    video_embedding = embeddings[ModalityType.VISION]
                    batch_similarity = F.cosine_similarity(video_embedding, audio_embedding) * 10

                    # Write results
                    for video_path, similarity in zip(video_batch_paths, batch_similarity.tolist()):
                        video_name = os.path.basename(os.path.dirname(video_path))
                        writer.writerow([video_name, similarity])
                except Exception as e:
                    print(f"Error processing batch {i}: {e}")
                    continue
except Exception as e:
    print(f"Error writing to the output CSV file {output_csv}: {e}")
    exit(1)

print(f"Similarity scores have been saved to {output_csv}.")
print(f"Any missing files have been logged in {error_log_file}.")