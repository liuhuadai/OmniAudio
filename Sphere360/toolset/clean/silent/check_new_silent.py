import os
import asyncio
from pydub import AudioSegment
from tqdm import tqdm
import numpy as np

# Async load single audio file
async def load_audio_file(audio_path):
    try:
        return AudioSegment.from_file(audio_path)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None

# Detect if audio is silent
def is_silent(audio, silence_threshold=-35, chunk_size=20):
    silence_count = 0
    total_chunks = len(audio) // chunk_size

    for i in range(total_chunks):
        chunk = audio[i * chunk_size:(i + 1) * chunk_size]
        
        # Convert multi-channel audio to numpy array and process each channel
        channels = chunk.split_to_mono()  # Split audio into mono channels
        max_dbfs = float('-inf')  # Initialize max value as negative infinity

        for channel in channels:
            # Get dBFS amplitude for each channel
            max_dbfs = max(max_dbfs, channel.dBFS)
        
        # Consider silent if max dBFS is below threshold
        if max_dbfs < silence_threshold:
            silence_count += 1

    silence_ratio = silence_count / total_chunks
    return silence_ratio > 0.9  # Mark as silent if over 90% is silent

# Process all audio files in directory
async def process_directory(directory_path, output_file, silence_threshold=-35.0, chunk_size=20):
    audio_files = [f for f in os.listdir(directory_path) if f.endswith('.flac')]
    audio_paths = [os.path.join(directory_path, f) for f in audio_files]

    silent_files = []

    # Process files in batches
    batch_size = 16  # Process 16 files per batch
    for i in tqdm(range(0, len(audio_paths), batch_size), desc="Processing batches"):
        batch_audio_paths = audio_paths[i:i+batch_size]
        # Async load current batch
        audio_list = await asyncio.gather(*[load_audio_file(path) for path in batch_audio_paths])

        # Process each audio file
        for audio, audio_path in zip(audio_list, batch_audio_paths):
            if audio and is_silent(audio, silence_threshold, chunk_size):
                silent_files.append(os.path.basename(audio_path))

    # Write silent files to output
    with open(output_file, 'w') as out_file:
        for file_name in silent_files:
            out_file.write(f"{file_name}\n")


if __name__ == "__main__":
    # Set paths
    input_directory = ""
    output_txt_file = ""

    # Run async task
    asyncio.run(process_directory(input_directory, output_txt_file))
