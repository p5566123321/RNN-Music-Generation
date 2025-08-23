import torch
import numpy as np
from model import MusicRNN, MusicGRU
from data_preprocessing import MIDIPreprocessor
import mido
import argparse
import random
from paths import MODEL_FILE, VOCAB_FILE, DEFAULT_OUTPUT_FILE

class EnhancedMusicGenerator:
    def __init__(self, model_path, vocab_path, device='cpu'):
        self.device = device
        self.preprocessor = MIDIPreprocessor(sequence_length=200, enhanced_mode=True)
        self.preprocessor.load_vocabulary(vocab_path)
        
        checkpoint = torch.load(model_path, map_location=device)
        vocab_size = checkpoint['vocab_size']
        
        self.model = MusicRNN(vocab_size=vocab_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
    def generate_enhanced_sequence(self, seed_sequence=None, length=500, temperature=1.0):
        """Generate enhanced music sequence using trained model."""
        with torch.no_grad():
            if seed_sequence is None:
                # Create a default seed with some musical structure
                seed_sequence = [
                    self.preprocessor.special_tokens['NOTE_ON'],
                    self.preprocessor.token_to_int.get('PITCH_60', 0),
                    self.preprocessor.token_to_int.get('VEL_8', 0),
                    self.preprocessor.token_to_int.get('TIME_4', 0),
                    self.preprocessor.special_tokens['NOTE_OFF'],
                    self.preprocessor.token_to_int.get('PITCH_60', 0)
                ]
                
                # Pad or trim seed to sequence length
                while len(seed_sequence) < self.preprocessor.sequence_length:
                    seed_sequence.append(self.preprocessor.special_tokens['PAD'])
                seed_sequence = seed_sequence[:self.preprocessor.sequence_length]
            
            sequence = seed_sequence.copy()
            hidden = None
            
            for _ in range(length):
                input_seq = torch.tensor([sequence[-self.preprocessor.sequence_length:]], 
                                       dtype=torch.long).to(self.device)
                
                output, hidden = self.model(input_seq, hidden)
                
                logits = output[0, -1, :] / temperature
                probabilities = torch.softmax(logits, dim=0)
                
                next_token = torch.multinomial(probabilities, 1).item()
                sequence.append(next_token)
                
                # Stop if we hit end of track
                if next_token == self.preprocessor.special_tokens['END_OF_TRACK']:
                    break
                    
            return sequence
    
    def create_musical_seed(self, notes=[60, 62, 64], velocities=None, durations=None):
        """Create a musically structured seed sequence."""
        if velocities is None:
            velocities = [8] * len(notes)  # Medium velocity
        if durations is None:
            durations = [4] * len(notes)  # Quarter note duration
            
        seed = []
        for i, (note, vel, dur) in enumerate(zip(notes, velocities, durations)):
            # Note on
            seed.extend([
                self.preprocessor.special_tokens['NOTE_ON'],
                self.preprocessor.token_to_int.get(f'PITCH_{note}', 0),
                self.preprocessor.token_to_int.get(f'VEL_{vel}', 0)
            ])
            
            # Duration (time shift)
            if i < len(notes) - 1:  # Don't add time after last note
                seed.extend([
                    self.preprocessor.token_to_int.get(f'TIME_{dur}', 0)
                ])
            
            # Note off
            seed.extend([
                self.preprocessor.special_tokens['NOTE_OFF'],
                self.preprocessor.token_to_int.get(f'PITCH_{note}', 0)
            ])
        
        # Pad to sequence length
        while len(seed) < self.preprocessor.sequence_length:
            seed.append(self.preprocessor.special_tokens['PAD'])
            
        return seed[:self.preprocessor.sequence_length]
    
    def create_seed_from_midi(self, midi_file):
        """Create seed sequence from a MIDI file for enhanced mode."""
        import os
        if not os.path.exists(midi_file):
            raise FileNotFoundError(f"MIDI file not found: {midi_file}")
            
        # Temporarily store the current token mapping
        original_token_to_int = self.preprocessor.token_to_int.copy()
        original_int_to_token = self.preprocessor.int_to_token.copy()
        original_current_token_id = self.preprocessor.current_token_id
        
        try:
            # Process MIDI file using enhanced mode
            tokens = self.preprocessor.midi_to_notes(midi_file)
            
            # Filter tokens to only include those in the loaded vocabulary
            vocab_size = len(original_token_to_int)
            filtered_tokens = []
            
            for token in tokens:
                if token < vocab_size:
                    filtered_tokens.append(token)
                else:
                    # Replace unknown tokens with PAD
                    filtered_tokens.append(self.preprocessor.special_tokens['PAD'])
            
            # Return the last sequence_length tokens as seed
            if len(filtered_tokens) >= self.preprocessor.sequence_length:
                return filtered_tokens[-self.preprocessor.sequence_length:]
            else:
                # If MIDI is shorter than sequence_length, pad with PAD tokens
                seed = filtered_tokens + [self.preprocessor.special_tokens['PAD']] * (self.preprocessor.sequence_length - len(filtered_tokens))
                return seed[:self.preprocessor.sequence_length]
                
        finally:
            # Restore original token mapping to prevent vocab expansion
            self.preprocessor.token_to_int = original_token_to_int
            self.preprocessor.int_to_token = original_int_to_token
            self.preprocessor.current_token_id = original_current_token_id
    
    def generate_to_midi(self, output_file, length=500, temperature=1.0, 
                        seed_notes=None, seed_midi=None, bpm=120):
        """Generate enhanced sequence and convert directly to MIDI."""
        if seed_midi:
            seed_sequence = self.create_seed_from_midi(seed_midi)
        elif seed_notes:
            seed_sequence = self.create_musical_seed(seed_notes)
        else:
            seed_sequence = None
            
        generated_sequence = self.generate_enhanced_sequence(
            seed_sequence, length, temperature
        )
        
        self.preprocessor.enhanced_tokens_to_midi(
            generated_sequence, output_file, bpm
        )
        
        return generated_sequence

def main():
    parser = argparse.ArgumentParser(description='Generate music using enhanced RNN')
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
    parser.add_argument('--seed-midi', type=str, default=None,
                       help='Path to MIDI file to use as seed input')
    parser.add_argument('--bpm', type=int, default=120,
                       help='Beats per minute (BPM) for the generated music')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        generator = EnhancedMusicGenerator(args.model, args.vocab, device.type)
        
        if args.seed_midi:
            seed_notes = None
            seed_midi = args.seed_midi
            print(f"Using MIDI seed from: {args.seed_midi}")
        elif args.seed:
            seed_notes = [int(note) for note in args.seed.split(',')]
            seed_midi = None
            print(f"Using seed notes: {seed_notes}")
        else:
            seed_notes = None
            seed_midi = None
            print("Using default musical seed")
        
        print(f"Generating {args.length} tokens with temperature {args.temperature}...")
        print(f"Enhanced mode vocabulary size: {generator.preprocessor.vocab_size}")
        
        generator.generate_to_midi(
            args.output, args.length, args.temperature, 
            seed_notes, seed_midi, args.bpm
        )
        
        print("Enhanced music generation completed!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a model trained with enhanced mode tokenization.")

if __name__ == "__main__":
    main()