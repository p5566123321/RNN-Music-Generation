import torch
import numpy as np
from model import MusicRNN, MusicGRU
from data_preprocessing import MIDIPreprocessor
import mido
import argparse
import random
from paths import MODEL_FILE, VOCAB_FILE, DEFAULT_OUTPUT_FILE

class MusicGenerator:
    def __init__(self, model_path, vocab_path, device='cpu'):
        self.device = device
        self.preprocessor = MIDIPreprocessor()
        self.preprocessor.load_vocabulary(vocab_path)
        
        checkpoint = torch.load(model_path, map_location=device)
        vocab_size = checkpoint['vocab_size']
        
        self.model = MusicRNN(vocab_size=vocab_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
    def generate_sequence(self, seed_sequence, length=500, temperature=1.0):
        """Generate music sequence using trained model."""
        with torch.no_grad():
            sequence = seed_sequence.copy()
            hidden = None
            
            for _ in range(length):
                input_seq = torch.tensor([sequence[-self.preprocessor.sequence_length:]], 
                                       dtype=torch.long).to(self.device)
                
                output, hidden = self.model(input_seq, hidden)
                
                logits = output[0, -1, :] / temperature
                probabilities = torch.softmax(logits, dim=0)
                
                next_note = torch.multinomial(probabilities, 1).item()
                sequence.append(next_note)
                
            return sequence
    
    def create_seed_from_notes(self, notes):
        """Create seed sequence from list of note numbers."""
        int_notes = [self.preprocessor.note_to_int.get(note, 0) for note in notes]
        return int_notes[-self.preprocessor.sequence_length:]
    
    def random_seed(self):
        """Create random seed sequence."""
        return [random.randint(0, self.preprocessor.vocab_size - 1) 
                for _ in range(self.preprocessor.sequence_length)]
    
    def sequence_to_midi(self, sequence, output_file, bpm=120):
        """Convert note sequence to MIDI file."""
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        
        # Convert BPM to MIDI tempo (microseconds per quarter note)
        tempo = int(60000000 / bpm)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo))
        
        time_per_note = 480
        
        for note_int in sequence:
            if note_int in self.preprocessor.int_to_note:
                note_data = self.preprocessor.int_to_note[note_int]
                
                # Handle both simple notes (int) and pitch-duration tuples
                if isinstance(note_data, tuple):
                    pitch, duration = note_data
                    note_duration = duration if duration > 0 else time_per_note
                else:
                    pitch = note_data
                    note_duration = time_per_note
                
                # Ensure pitch is within valid MIDI range
                if 0 <= pitch <= 127:
                    track.append(mido.Message('note_on', channel=0, 
                                            note=pitch, velocity=64, time=0))
                    track.append(mido.Message('note_off', channel=0, 
                                            note=pitch, velocity=64, time=note_duration))
        
        mid.save(output_file)
        print(f"MIDI file saved as: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate music using trained RNN')
    parser.add_argument('--model', type=str, default=MODEL_FILE,
                       help='Path to trained model')
    parser.add_argument('--vocab', type=str, default=VOCAB_FILE,
                       help='Path to vocabulary file')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_FILE,
                       help='Output MIDI file name')
    parser.add_argument('--length', type=int, default=500,
                       help='Length of generated sequence')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature (higher = more random)')
    parser.add_argument('--seed', type=str, default=None,
                       help='Comma-separated seed notes (e.g., "60,62,64")')
    parser.add_argument('--bpm', type=int, default=120,
                       help='Beats per minute (BPM) for the generated music')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    generator = MusicGenerator(args.model, args.vocab, device.type)
    
    if args.seed:
        seed_notes = [int(note) for note in args.seed.split(',')]
        seed_sequence = generator.create_seed_from_notes(seed_notes)
        print(f"Using seed notes: {seed_notes}")
    else:
        seed_sequence = generator.random_seed()
        print("Using random seed")
    
    print(f"Generating {args.length} notes with temperature {args.temperature}...")
    generated_sequence = generator.generate_sequence(
        seed_sequence, args.length, args.temperature
    )
    
    generator.sequence_to_midi(generated_sequence, args.output, args.bpm)
    print("Music generation completed!")

if __name__ == "__main__":
    main()