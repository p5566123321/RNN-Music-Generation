import torch
import torch.nn as nn

class PolyphonicMusicRNN(nn.Module):
    """RNN model for polyphonic music generation with multiple tracks."""
    
    def __init__(self, vocab_size: int, num_tracks: int = 4, embedding_dim: int = 128, 
                 hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.3):
        super(PolyphonicMusicRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.num_tracks = num_tracks
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Separate embedding for each track
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embedding_dim) for _ in range(num_tracks)
        ])
        
        # Shared LSTM for all tracks
        self.lstm = nn.LSTM(embedding_dim * num_tracks, hidden_dim, num_layers, 
                           dropout=dropout, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        
        # Separate output layer for each track
        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_dim, vocab_size) for _ in range(num_tracks)
        ])
        
    def forward(self, x, hidden=None):
        """
        x: [batch_size, seq_len, num_tracks]
        """
        batch_size, seq_len, num_tracks = x.size()
        
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        # Embed each track separately then concatenate
        embedded_tracks = []
        for i in range(num_tracks):
            embedded = self.embeddings[i](x[:, :, i])  # [batch_size, seq_len, embedding_dim]
            embedded_tracks.append(embedded)
        
        # Concatenate all track embeddings
        embedded = torch.cat(embedded_tracks, dim=-1)  # [batch_size, seq_len, embedding_dim * num_tracks]
        
        lstm_out, hidden = self.lstm(embedded, hidden)
        lstm_out = self.dropout(lstm_out)
        
        # Generate output for each track
        outputs = []
        for i in range(num_tracks):
            track_output = self.output_layers[i](lstm_out)  # [batch_size, seq_len, vocab_size]
            outputs.append(track_output)
        
        # Stack outputs: [batch_size, seq_len, num_tracks, vocab_size]
        output = torch.stack(outputs, dim=2)
        
        return output, hidden
    
    def init_hidden(self, batch_size: int, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)

class ChordBasedMusicRNN(nn.Module):
    """Alternative approach: Predict chord progressions."""
    
    def __init__(self, vocab_size: int, max_chord_size: int = 6, 
                 embedding_dim: int = 128, hidden_dim: int = 256, 
                 num_layers: int = 2, dropout: float = 0.3):
        super(ChordBasedMusicRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.max_chord_size = max_chord_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim * max_chord_size, hidden_dim, num_layers,
                           dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
        # Output layer predicts multiple notes simultaneously
        self.fc = nn.Linear(hidden_dim, vocab_size * max_chord_size)
        
    def forward(self, x, hidden=None):
        """
        x: [batch_size, seq_len, max_chord_size] - padded with special tokens for shorter chords
        """
        batch_size, seq_len, chord_size = x.size()
        
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        # Embed and flatten chord dimensions
        embedded = self.embedding(x)  # [batch_size, seq_len, chord_size, embedding_dim]
        embedded = embedded.view(batch_size, seq_len, -1)  # [batch_size, seq_len, chord_size * embedding_dim]
        
        lstm_out, hidden = self.lstm(embedded, hidden)
        lstm_out = self.dropout(lstm_out)
        
        output = self.fc(lstm_out)  # [batch_size, seq_len, vocab_size * max_chord_size]
        output = output.view(batch_size, seq_len, self.max_chord_size, self.vocab_size)
        
        return output, hidden
    
    def init_hidden(self, batch_size: int, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)