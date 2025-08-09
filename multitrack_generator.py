import torch
import mido
import random
from polyphonic_model import PolyphonicMusicRNN
from data_preprocessing import MIDIPreprocessor
from paths import MODEL_FILE, VOCAB_FILE, DEFAULT_OUTPUT_FILE

class MultiTrackGenerator:
    """Generate multi-track MIDI music."""
    
    def __init__(self, model_path, vocab_path, num_tracks=4, device='cpu'):
        self.device = device
        self.num_tracks = num_tracks
        self.preprocessor = MIDIPreprocessor()
        self.preprocessor.load_vocabulary(vocab_path)
        
        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        vocab_size = checkpoint['vocab_size']
        
        self.model = PolyphonicMusicRNN(
            vocab_size=vocab_size, 
            num_tracks=num_tracks,
            embedding_dim=128,
            hidden_dim=256,
            num_layers=2
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
    
    def generate_multitrack_sequence(self, seed_sequences, length=500, temperature=1.0):
        """
        Generate multi-track music sequence.
        seed_sequences: list of sequences for each track
        """
        with torch.no_grad():
            sequences = [seq.copy() for seq in seed_sequences]
            hidden = None
            
            for _ in range(length):
                # Prepare input for all tracks
                input_tracks = []
                for i, seq in enumerate(sequences):
                    track_input = seq[-self.preprocessor.sequence_length:]
                    # Pad if too short
                    while len(track_input) < self.preprocessor.sequence_length:
                        track_input.insert(0, 0)  # Pad with rest/silence token
                    input_tracks.append(track_input)
                
                # Shape: [1, seq_len, num_tracks]
                input_tensor = torch.tensor([list(zip(*input_tracks))], 
                                          dtype=torch.long).to(self.device)
                input_tensor = input_tensor.transpose(1, 2)  # [1, num_tracks, seq_len]
                input_tensor = input_tensor.transpose(1, 2)  # [1, seq_len, num_tracks]
                
                output, hidden = self.model(input_tensor, hidden)
                
                # Sample next note for each track
                for track_idx in range(self.num_tracks):
                    logits = output[0, -1, track_idx, :] / temperature
                    probabilities = torch.softmax(logits, dim=0)
                    next_note = torch.multinomial(probabilities, 1).item()
                    sequences[track_idx].append(next_note)
            
            return sequences
    
    def create_multitrack_midi(self, sequences, output_file, tempo=500000):
        """Convert multi-track sequences to MIDI file."""
        mid = mido.MidiFile()
        
        # Create separate track for each voice
        track_names = ['Piano', 'Bass', 'Melody', 'Harmony']
        channels = [0, 1, 2, 3]
        
        for track_idx, sequence in enumerate(sequences):
            track = mido.MidiTrack()
            mid.tracks.append(track)
            
            # Add track name
            if track_idx < len(track_names):
                track.append(mido.MetaMessage('track_name', name=track_names[track_idx]))
            
            # Set tempo (only on first track)
            if track_idx == 0:
                track.append(mido.MetaMessage('set_tempo', tempo=tempo))
            
            # Convert sequence to MIDI messages
            time_per_note = 480
            channel = channels[track_idx] if track_idx < len(channels) else 0
            
            for note_int in sequence:
                if note_int in self.preprocessor.int_to_note and note_int != 0:  # Skip rest tokens
                    note = self.preprocessor.int_to_note[note_int]
                    
                    # Ensure note is in valid MIDI range
                    if 0 <= note <= 127:
                        track.append(mido.Message('note_on', channel=channel,
                                                note=note, velocity=64, time=0))
                        track.append(mido.Message('note_off', channel=channel,
                                                note=note, velocity=64, time=time_per_note))
                else:
                    # Rest - just advance time
                    track.append(mido.Message('note_off', channel=channel, 
                                              note=0, velocity=0, time=time_per_note))
        
        mid.save(output_file)
        print(f"Multi-track MIDI saved as: {output_file}")

def create_simple_multitrack_from_mono(mono_model_path, vocab_path, output_file=DEFAULT_OUTPUT_FILE, bpm=75):
    """
    Simple approach: Generate multiple monophonic tracks using the existing model.
    """
    from generate import MusicGenerator
    
    # Generate 4 different tracks with different seeds and temperatures
    track_configs = [
        {'seed': [60, 62, 64, 65], 'temperature': 0.8, 'length': 400},  # Melody
        {'seed': [48, 50, 52, 53], 'temperature': 0.6, 'length': 400},  # Bass
        {'seed': [67, 69, 71, 72], 'temperature': 0.9, 'length': 400},  # Harmony
        {'seed': [72, 74, 76, 77], 'temperature': 1.0, 'length': 400},  # High melody
    ]
    
    generator = MusicGenerator(mono_model_path, vocab_path)
    
    # Create MIDI file with multiple tracks
    mid = mido.MidiFile()
    track_names = ['Melody', 'Bass', 'Harmony', 'High Melody']
    
    for i, config in enumerate(track_configs):
        # Generate sequence for this track
        seed_seq = generator.create_seed_from_notes(config['seed'])
        sequence = generator.generate_sequence(
            seed_seq, config['length'], config['temperature']
        )
        
        # Create MIDI track
        track = mido.MidiTrack()
        mid.tracks.append(track)
        tempo = int(60000000 / bpm)
        
        track.append(mido.MetaMessage('track_name', name=track_names[i]))
        if i == 0:  # Set tempo on first track
            track.append(mido.MetaMessage('set_tempo', tempo=tempo))
        
        # Convert to MIDI messages
        time_per_note = 480
        channel = i  # Use different channel for each track
        
        for note_int in sequence:
            if note_int in generator.preprocessor.int_to_note:
                note = generator.preprocessor.int_to_note[note_int]
                if 0 <= note <= 127:
                    track.append(mido.Message('note_on', channel=channel,
                                            note=note, velocity=64, time=0))
                    track.append(mido.Message('note_off', channel=channel,
                                            note=note, velocity=64, time=time_per_note))
    
    mid.save(output_file)
    print(f"Multi-track MIDI saved as: {output_file}")

if __name__ == "__main__":
    # Use the simple approach with existing monophonic model
    create_simple_multitrack_from_mono(
        MODEL_FILE , 
        VOCAB_FILE ,
        DEFAULT_OUTPUT_FILE,
    )