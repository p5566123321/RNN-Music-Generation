import os
import mido
from data_preprocessing import MIDIPreprocessor
from paths import MIDI_DIR
def check_midi_files(midi_dir=MIDI_DIR):
    """Check MIDI files in directory for compatibility."""
    if not os.path.exists(midi_dir):
        print(f"Directory {midi_dir} doesn't exist!")
        return
    
    midi_files = [f for f in os.listdir(midi_dir) if f.endswith(('.mid', '.midi'))]
    
    if not midi_files:
        print(f"No MIDI files found in {midi_dir}")
        return
    
    print(f"Found {len(midi_files)} MIDI files:")
    
    preprocessor = MIDIPreprocessor()
    total_notes = 0
    
    for file in midi_files:
        try:
            filepath = os.path.join(midi_dir, file)
            notes = preprocessor.midi_to_notes(filepath)
            total_notes += len(notes)
            print(f"✓ {file}: {len(notes)} notes")
        except Exception as e:
            print(f"✗ {file}: Error - {e}")
    
    print(f"\nTotal notes: {total_notes}")
    print(f"Estimated training sequences: {max(0, total_notes - 100)}")

if __name__ == "__main__":
    check_midi_files()