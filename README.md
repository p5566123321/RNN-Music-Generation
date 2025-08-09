# RNN Music Generation with PyTorch

A deep learning project for generating music using Recurrent Neural Networks (LSTM/GRU) implemented in PyTorch.

## Features

- MIDI file preprocessing and data preparation
- LSTM and GRU model architectures
- Training pipeline with gradient clipping and learning rate scheduling
- Music generation with controllable randomness (temperature sampling)
- MIDI output functionality

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your MIDI dataset:
   - Create a `midi_files/` directory
   - Add your MIDI files (.mid or .midi) to this directory

## Usage

### Training

Train the model on your MIDI dataset:

```bash
python train.py
```

This will:
- Preprocess all MIDI files in `midi_files/`
- Create vocabulary mappings
- Train the RNN model
- Save the trained model as `music_rnn_model.pth`
- Save vocabulary as `vocabulary.pkl`

### Generation

Generate new music using the trained model:

```bash
# Basic generation
python generate.py

# With custom parameters
python generate.py --length 1000 --temperature 0.8 --output my_song.mid

# With seed notes (middle C, D, E)
python generate.py --seed "60,62,64" --length 500
```

## Parameters

### Training Parameters
- `sequence_length`: Length of input sequences (default: 100)
- `embedding_dim`: Embedding dimension (default: 128)
- `hidden_dim`: RNN hidden dimension (default: 256)
- `num_layers`: Number of RNN layers (default: 2)
- `dropout`: Dropout rate (default: 0.3)
- `epochs`: Number of training epochs (default: 50)
- `batch_size`: Training batch size (default: 64)
- `learning_rate`: Learning rate (default: 0.001)

### Generation Parameters
- `length`: Number of notes to generate (default: 500)
- `temperature`: Sampling temperature - higher values = more randomness (default: 1.0)
- `seed`: Starting notes for generation (optional)

## Model Architecture

The project includes two RNN variants:

1. **MusicRNN**: LSTM-based model
2. **MusicGRU**: GRU-based model

Both models use:
- Embedding layer for note representation
- Multi-layer RNN with dropout
- Linear output layer for note prediction

## Files Structure

- `data_preprocessing.py`: MIDI data preprocessing utilities
- `model.py`: RNN model architectures (LSTM/GRU)
- `train.py`: Training script and data loading
- `generate.py`: Music generation and MIDI output
- `requirements.txt`: Python dependencies

## Tips

- Use temperature values between 0.5-1.5 for best results
- Lower temperature (0.5-0.8) = more predictable, coherent music
- Higher temperature (1.2-1.5) = more creative, random music
- Train on similar style MIDI files for consistent generation
- Larger datasets and longer training generally produce better results