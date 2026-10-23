from pathlib import Path
import shutil
import librosa

# Find the main project folder.
BASE_DIR = Path(__file__).resolve().parent.parent

# Set the folder where sample audio files will be saved.
SAMPLE_DIR = BASE_DIR / "sample_audio"

# Create the folder if it doesn't already exist.
SAMPLE_DIR.mkdir(exist_ok=True)

# Download an example audio recording from Librosa.
audio_path = Path(librosa.ex("brahms"))

# Name the downloaded file sample_music_2.
# Keep the original audio format (.ogg, .mp3, etc.).
destination = SAMPLE_DIR / f"sample_music_2{audio_path.suffix}"

# Copy the audio file into the sample_audio folder.
shutil.copy(audio_path, destination)

print(f"Sample audio saved successfully: {destination}")