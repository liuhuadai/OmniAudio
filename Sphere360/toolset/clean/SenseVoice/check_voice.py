import os
import csv
from tqdm import tqdm  # For progress bar display
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# Define model path
model_dir = "iic/SenseVoiceSmall"

# Initialize model
model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    remote_code="./model.py",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
)

# Define audio folder path
audio_folder = ""

# Output CSV file path
output_csv = "./recognition_results.csv"

# Get all .flac files in audio folder
audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".flac")]

# Prepare CSV file and write header (if file is empty)
if not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0:
    with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Audio File", "Transcription"])  # CSV column headers

# Get existing processed audio files to avoid reprocessing
existing_files = set()
with open(output_csv, mode="r", newline="", encoding="utf-8") as file:
    reader = csv.reader(file)
    next(reader)  # Skip header row
    for row in reader:
        existing_files.add(row[0])  # Add processed files to set

# Process all .flac files in audio folder
with open(output_csv, mode="a", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    # Show progress bar using tqdm
    for audio_file in tqdm(audio_files, desc="Processing", unit="file"):
        # Skip if file already processed
        if audio_file in existing_files:
            continue

        audio_path = os.path.join(audio_folder, audio_file)

        try:
            # Perform speech recognition
            res = model.generate(
                input=audio_path,
                cache={},
                language="auto",  # Auto-detect language
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )

            # Get transcription with post-processing
            transcription = rich_transcription_postprocess(res[0]["text"])

            # Mark as "none!" if transcription is empty
            if not transcription.strip():
                transcription = "none!"

        except Exception as e:
            # Record error if recognition fails
            transcription = f"Error: {str(e)}"
        
        # Write filename and transcription to CSV
        writer.writerow([audio_file, transcription])

print("Recognition completed and saved to CSV.")