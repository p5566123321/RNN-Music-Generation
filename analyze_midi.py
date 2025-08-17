import mido
import sys
from data_preprocessing import MIDIPreprocessor

def analyze_midi_file(midi_file):
    """Analyze a MIDI file to understand its structure."""
    print(f"\n=== Analyzing {midi_file} ===")
    
    try:
        mid = mido.MidiFile(midi_file)
        print(f"MIDI Type: {mid.type}")
        print(f"Ticks per beat: {mid.ticks_per_beat}")
        print(f"Number of tracks: {len(mid.tracks)}")
        print(f"Length in seconds: {mid.length:.2f}")
        
        total_messages = 0
        note_messages = 0
        unique_notes = set()
        
        for i, track in enumerate(mid.tracks):
            track_notes = 0
            print(f"\nTrack {i} ({len(track)} messages):")
            
            for msg in track:
                total_messages += 1
                if msg.type == 'note_on' and msg.velocity > 0:
                    note_messages += 1
                    track_notes += 1
                    unique_notes.add(msg.note)
                elif msg.type in ['track_name', 'program_change']:
                    print(f"  {msg}")
            
            print(f"  Note events: {track_notes}")
        
        print(f"\nSummary:")
        print(f"Total messages: {total_messages}")
        print(f"Note on messages: {note_messages}")
        print(f"Unique notes: {len(unique_notes)} (range: {min(unique_notes) if unique_notes else 'N/A'} to {max(unique_notes) if unique_notes else 'N/A'})")
        
        # Test both modes
        print(f"\n=== Tokenization Analysis ===")
        
        # Simple mode
        preprocessor_simple = MIDIPreprocessor(enhanced_mode=False)
        simple_notes = preprocessor_simple.midi_to_notes(midi_file)
        print(f"Simple mode tokens: {len(simple_notes)}")
        
        # Enhanced mode  
        preprocessor_enhanced = MIDIPreprocessor(enhanced_mode=True)
        enhanced_tokens = preprocessor_enhanced.midi_to_notes(midi_file)
        print(f"Enhanced mode tokens: {len(enhanced_tokens)}")
        
        if len(enhanced_tokens) > 0:
            preprocessor_enhanced.create_vocabulary(enhanced_tokens)
            print(f"Enhanced vocabulary size: {preprocessor_enhanced.vocab_size}")
            
            # Show some example tokens
            print(f"First 20 tokens: {enhanced_tokens[:20]}")
            token_names = [preprocessor_enhanced.int_to_token.get(t, f"UNK_{t}") for t in enhanced_tokens[:10]]
            print(f"Token names: {token_names}")
        
    except Exception as e:
        print(f"Error analyzing {midi_file}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_midi.py <midi_file>")
    else:
        analyze_midi_file(sys.argv[1])