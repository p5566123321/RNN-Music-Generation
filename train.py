import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_preprocessing import MIDIPreprocessor
from model import MusicRNN
from paths import MODEL_FILE, VOCAB_FILE, DEFAULT_OUTPUT_FILE, MIDI_DIR

class MusicDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MusicTrainer:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.losses = []
        
    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y in tqdm(dataloader, desc="Training"):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            optimizer.zero_grad()
            
            output, _ = self.model(batch_x)
            loss = criterion(output[:, -1, :], batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
    
    def train(self, dataloader, epochs=50, lr=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
        
        print(f"Training on {self.device}")
        
        for epoch in range(epochs):
            avg_loss = self.train_epoch(dataloader, optimizer, criterion)
            self.losses.append(avg_loss)
            scheduler.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
                
        print("Training completed!")
        
    def save_model(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab_size': self.model.vocab_size,
            'losses': self.losses
        }, filepath)
        
    def plot_losses(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    preprocessor = MIDIPreprocessor(sequence_length=100)
    
    X, y = preprocessor.preprocess_dataset(MIDI_DIR)
    
    preprocessor.save_vocabulary(VOCAB_FILE)
    
    dataset = MusicDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = MusicRNN(vocab_size=preprocessor.vocab_size, 
                     embedding_dim=128, hidden_dim=256, num_layers=2)

    trainer = MusicTrainer(model, str(device))
    trainer.train(dataloader, epochs=10, lr=0.0001)
    
    trainer.save_model(MODEL_FILE)
    trainer.plot_losses()

if __name__ == "__main__":
    main()