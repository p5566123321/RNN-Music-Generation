import mido
import numpy as np
import os
from typing import List, Tuple, Dict
import pickle

class MIDIPreprocessor:
    def __init__(self, sequence_length: int = 100, enhanced_mode: bool = False):
        self.sequence_length = sequence_length
        self.enhanced_mode = enhanced_mode
        self.note_to_int = {}
        self.int_to_note = {}
        self.vocab_size = 0
        
        # Enhanced mode tokens
        if enhanced_mode:
            self.special_tokens = {
                'PAD': 0,
                'NOTE_ON': 1,
                'NOTE_OFF': 2, 
                'TIME_SHIFT': 3,
                'VELOCITY': 4,
                'PROGRAM_CHANGE': 5,
                'END_OF_TRACK': 6
            }
            self.token_to_int = self.special_tokens.copy()
            self.int_to_token = {v: k for k, v in self.special_tokens.items()}
            self.current_token_id = len(self.special_tokens)
        
    def midi_to_notes(self, midi_file: str) -> List:
        """Convert MIDI file to list of note numbers (legacy mode)."""
        if self.enhanced_mode:
            return self.midi_to_enhanced_tokens(midi_file)

        return self.midi_to_pitch_duration(midi_file)
    
    def midi_to_pitch_duration(self, midi_file: str) -> List[Tuple[int, int]]:
        """
        Convert a MIDI file into a list of (pitch, duration) tuples.
        - pitch: MIDI note number (0-127)
        - duration: time in ticks (relative length)
        """
        mid = mido.MidiFile(midi_file)
        notes = []

        # Keep track of notes that are currently "on"
        active_notes = {}   # note -> start_time
        current_time = 0

        for track in mid.tracks:
            current_time = 0
            for msg in track:
                current_time += msg.time  # accumulate delta times
                if msg.type == 'note_on' and msg.velocity > 0:
                    # Note pressed
                    active_notes[msg.note] = current_time
                elif msg.type in ('note_off', 'note_on') and msg.note in active_notes:
                    # Note released (explicit note_off or note_on with velocity=0)
                    start_time = active_notes.pop(msg.note)
                    duration = current_time - start_time
                    if duration > 0:
                        notes.append((msg.note, duration))
        
        return notes
    
    def midi_to_enhanced_tokens(self, midi_file: str) -> List[int]:
        """Convert MIDI file to enhanced token sequence."""
        mid = mido.MidiFile(midi_file)
        tokens = []
        
        for track in mid.tracks:
            current_time = 0
            
            for msg in track:
                # Add time shift tokens if needed
                if msg.time > 0:
                    # Quantize time to reduce vocabulary size
                    time_units = min(msg.time // 120, 127)  # Max 127 time units
                    if time_units > 0:
                        time_token = self._get_or_create_token(f"TIME_{time_units}")
                        tokens.append(time_token)
                
                if msg.type == 'note_on':
                    if msg.velocity > 0:
                        # Note on event
                        tokens.append(self.special_tokens['NOTE_ON'])
                        note_token = self._get_or_create_token(f"PITCH_{msg.note}")
                        tokens.append(note_token)
                        
                        # Velocity (quantized to reduce vocab size)
                        vel_quantized = msg.velocity // 8  # 16 velocity levels
                        vel_token = self._get_or_create_token(f"VEL_{vel_quantized}")
                        tokens.append(vel_token)
                    else:
                        # Note off (velocity 0)
                        tokens.append(self.special_tokens['NOTE_OFF'])
                        note_token = self._get_or_create_token(f"PITCH_{msg.note}")
                        tokens.append(note_token)
                        
                elif msg.type == 'note_off':
                    tokens.append(self.special_tokens['NOTE_OFF'])
                    note_token = self._get_or_create_token(f"PITCH_{msg.note}")
                    tokens.append(note_token)
                    
                elif msg.type == 'program_change':
                    tokens.append(self.special_tokens['PROGRAM_CHANGE'])
                    prog_token = self._get_or_create_token(f"PROG_{msg.program}")
                    tokens.append(prog_token)
            
            tokens.append(self.special_tokens['END_OF_TRACK'])
        
        return tokens
    
    def _get_or_create_token(self, token_name: str) -> int:
        """Get existing token ID or create new one."""
        if token_name in self.token_to_int:
            return self.token_to_int[token_name]
        
        token_id = self.current_token_id
        self.token_to_int[token_name] = token_id
        self.int_to_token[token_id] = token_name
        self.current_token_id += 1
        return token_id
    
    def create_vocabulary(self, notes: List[int]) -> None:
        """Create note-to-int and int-to-note mappings."""
        if self.enhanced_mode:
            self.vocab_size = self.current_token_id
            # For enhanced mode, vocab is already built in token_to_int
            self.note_to_int = self.token_to_int.copy()
            self.int_to_note = self.int_to_token.copy()
        else:
            unique_notes = sorted(set(notes))
            self.vocab_size = len(unique_notes)
            
            self.note_to_int = {note: i for i, note in enumerate(unique_notes)}
            self.int_to_note = {i: note for i, note in enumerate(unique_notes)}
    
    def notes_to_sequences(self, notes: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert notes to input-output sequences for training."""
        if self.enhanced_mode:
            # In enhanced mode, tokens are already integers
            int_notes = notes
        else:
            # Create vocabulary if it doesn't exist
            if not self.note_to_int:
                self.create_vocabulary(notes)
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
            'vocab_size': self.vocab_size,
            'enhanced_mode': self.enhanced_mode,
            'sequence_length': self.sequence_length
        }
        if self.enhanced_mode:
            vocab_data.update({
                'token_to_int': self.token_to_int,
                'int_to_token': self.int_to_token,
                'special_tokens': self.special_tokens,
                'current_token_id': self.current_token_id
            })
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(vocab_data, f)
    
    def load_vocabulary(self, filepath: str) -> None:
        """Load vocabulary mappings."""
        with open(filepath, 'rb') as f:
            vocab_data = pickle.load(f)
        self.note_to_int = vocab_data['note_to_int']
        self.int_to_note = vocab_data['int_to_note']
        self.vocab_size = vocab_data['vocab_size']
        self.enhanced_mode = vocab_data.get('enhanced_mode', False)
        self.sequence_length = vocab_data.get('sequence_length', self.sequence_length)
        
        if self.enhanced_mode:
            self.token_to_int = vocab_data['token_to_int']
            self.int_to_token = vocab_data['int_to_token']
            self.special_tokens = vocab_data['special_tokens']
            self.current_token_id = vocab_data['current_token_id']
    
    def enhanced_tokens_to_midi(self, tokens: List[int], output_file: str, bpm: int = 120):
        """Convert enhanced token sequence back to MIDI file."""
        if not self.enhanced_mode:
            raise ValueError("Enhanced tokens to MIDI conversion requires enhanced mode")
        
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        
        # Set tempo
        tempo = int(60000000 / bpm)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo))
        
        current_time = 0
        note_states = {}  # Track which notes are currently on
        
        i = 0
        while i < len(tokens):
            token_id = tokens[i]
            token_name = self.int_to_token.get(token_id, "UNKNOWN")
            
            if token_name.startswith('TIME_'):
                time_units = int(token_name.split('_')[1])
                current_time += time_units * 120  # Convert back from quantized time
                i += 1
                
            elif token_id == self.special_tokens['NOTE_ON'] and i + 2 < len(tokens):
                pitch_token = self.int_to_token.get(tokens[i + 1], "")
                vel_token = self.int_to_token.get(tokens[i + 2], "")
                
                if pitch_token.startswith('PITCH_') and vel_token.startswith('VEL_'):
                    pitch = int(pitch_token.split('_')[1])
                    vel_level = int(vel_token.split('_')[1])
                    velocity = min(127, vel_level * 8 + 4)  # Convert back from quantized
                    
                    track.append(mido.Message('note_on', channel=0, note=pitch, 
                                            velocity=velocity, time=current_time))
                    note_states[pitch] = True
                    current_time = 0
                    i += 3
                else:
                    i += 1
                    
            elif token_id == self.special_tokens['NOTE_OFF'] and i + 1 < len(tokens):
                pitch_token = self.int_to_token.get(tokens[i + 1], "")
                
                if pitch_token.startswith('PITCH_'):
                    pitch = int(pitch_token.split('_')[1])
                    
                    track.append(mido.Message('note_off', channel=0, note=pitch, 
                                            velocity=64, time=current_time))
                    note_states.pop(pitch, None)
                    current_time = 0
                    i += 2
                else:
                    i += 1
                    
            elif token_id == self.special_tokens['PROGRAM_CHANGE'] and i + 1 < len(tokens):
                prog_token = self.int_to_token.get(tokens[i + 1], "")
                if prog_token.startswith('PROG_'):
                    program = int(prog_token.split('_')[1])
                    track.append(mido.Message('program_change', channel=0, program=program, time=current_time))
                    current_time = 0
                    i += 2
                else:
                    i += 1
                    
            elif token_id == self.special_tokens['END_OF_TRACK']:
                # Turn off any remaining notes
                for pitch in list(note_states.keys()):
                    track.append(mido.Message('note_off', channel=0, note=pitch, velocity=64, time=0))
                break
            else:
                i += 1
        
        mid.save(output_file)
        print(f"Enhanced MIDI file saved as: {output_file}")