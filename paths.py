from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Paths to specific directories or files
MODEL_DIR = BASE_DIR / "model"
VOCAB_DIR = BASE_DIR / "vocab"
OUTPUT_DIR = BASE_DIR / "output"
MIDI_DIR = str(BASE_DIR / "midi_files")

# Specific files  
MODEL_FILE = str(MODEL_DIR / "music_rnn_model.pth")
VOCAB_FILE = str(VOCAB_DIR / "vocabulary.pkl")
DEFAULT_OUTPUT_FILE = str(OUTPUT_DIR / "generated_music.mid")