
#!/usr/bin/env python3
"""
Train HuBERT encoder to predict WavLM aesthetics scores (knowledge distillation)
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import pandas as pd
import torchaudio
from tqdm import tqdm
from pathlib import Path

print("=" * 70)
print("TRAINING HUBERT STUDENT MODEL")
print("=" * 70)

# Config
LABELS_CSV = "data/audio_labels.csv"
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
DEVICE = 'cuda' if torch.cuda. is_available() else 'cpu'
OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"\n📋 Config:")
print(f"   Device: {DEVICE}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Epochs: {EPOCHS}")
print(f"   Learning rate: {LR}")

# Dataset
class AudioDataset(Dataset):
    def __init__(self, csv_path, feature_extractor):
        self.df = pd.read_csv(csv_path)
        self.feature_extractor = feature_extractor
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load audio
        waveform, sr = torchaudio.load(row['filepath'])
        
        # Resample to 16kHz if needed
        if sr != 16000:
            resampler = torchaudio. transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Extract features
        inputs = self.feature_extractor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        # Target labels (from WavLM teacher)
        labels = torch.tensor([
            row['CE'], row['CU'], row['PC'], row['PQ']
        ], dtype=torch. float32)
        
        return inputs['input_values']. squeeze(), labels

# Model
class HuBERTAestheticsPredictor(nn. Module):
    def __init__(self):
        super().__init__()
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        
        # Freeze first 6 layers (optional - faster training)
        for param in self.hubert.encoder.layers[:6].parameters():
            param.requires_grad = False
        
        # Regression head
        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 4)  # CE, CU, PC, PQ
        )
    
    def forward(self, input_values):
        outputs = self.hubert(input_values)
        # Pool over time dimension
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.head(pooled)

# Training
print(f"\n[1] Loading dataset...")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
dataset = AudioDataset(LABELS_CSV, feature_extractor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
print(f"    ✅ Loaded {len(dataset)} samples")

print(f"\n[2] Initializing model...")
model = HuBERTAestheticsPredictor().to(DEVICE)
optimizer = torch.optim.AdamW(model. parameters(), lr=LR)
criterion = nn.MSELoss()
print(f"    ✅ Model ready")

print(f"\n[3] Training for {EPOCHS} epochs.. .\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for inputs, labels in pbar:
        inputs = inputs. to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss':  f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    print(f"   Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
    
    # Save checkpoint
    checkpoint_path = OUTPUT_DIR / f"hubert_aesthetics_epoch{epoch+1}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, checkpoint_path)
    print(f"   💾 Saved:  {checkpoint_path}")

print("\n" + "=" * 70)
print("✅ TRAINING COMPLETE!")
print("=" * 70)
print(f"\nModel saved to: {OUTPUT_DIR}/")
print("\nNext: Test inference with the trained model")
