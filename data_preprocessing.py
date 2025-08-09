import mido
import numpy as np
import os
from typing import List, Tuple, Dict
import pickle

class MIDIPreprocessor:
    def __init__(self, sequence_length: int = 100):
        self.sequence_length = sequence_length
        self.note_to_int = {}
        self.int_to_note = {}
        self.vocab_size = 0
        
    def midi_to_notes(self, midi_file: str) -> List[int]:
        """Convert MIDI file to list of note numbers."""
        mid = mido.MidiFile(midi_file)
        notes = []
        
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    notes.append(msg.note)
                    
        return notes
    
    def create_vocabulary(self, notes: List[int]) -> None:
        """Create note-to-int and int-to-note mappings."""
        unique_notes = sorted(set(notes))
        self.vocab_size = len(unique_notes)
        
        self.note_to_int = {note: i for i, note in enumerate(unique_notes)}
        self.int_to_note = {i: note for i, note in enumerate(unique_notes)}
    
    def notes_to_sequences(self, notes: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert notes to input-output sequences for training."""
        int_notes = [self.note_to_int[note] for note in notes]
        
        X, y = [], []
        for i in range(len(int_notes) - self.sequence_length):
            X.append(int_notes[i:i + self.sequence_length])
            y.append(int_notes[i + self.sequence_length])
            
        return np.array(X), np.array(y)
    
    def preprocess_dataset(self, midi_dir) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess all MIDI files in directory."""
        all_notes = []
        
        for file in os.listdir(midi_dir):
            if file.endswith('.mid') or file.endswith('.midi'):
                try:
                    notes = self.midi_to_notes(os.path.join(midi_dir, file))
                    all_notes.extend(notes)
                except Exception as e:
                    print(f"Error processing {file}: {e}")
                    
        self.create_vocabulary(all_notes)
        return self.notes_to_sequences(all_notes)
    
    def save_vocabulary(self, filepath) -> None:
        """Save vocabulary mappings."""
        vocab_data = {
            'note_to_int': self.note_to_int,
            'int_to_note': self.int_to_note,
            'vocab_size': self.vocab_size
        }
        with open(filepath, 'wb') as f:
            pickle.dump(vocab_data, f)
    
    def load_vocabulary(self, filepath: str) -> None:
        """Load vocabulary mappings."""
        with open(filepath, 'rb') as f:
            vocab_data = pickle.load(f)
        self.note_to_int = vocab_data['note_to_int']
        self.int_to_note = vocab_data['int_to_note']
        self.vocab_size = vocab_data['vocab_size']