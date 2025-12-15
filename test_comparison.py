
#!/usr/bin/env python3
import torch
import torchaudio
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import torch.nn as nn
from pathlib import Path
import pandas as pd
from audiobox_aesthetics.infer import initialize_predictor

# Load HuBERT model
class HuBERTAestheticsPredictor(nn. Module):
    def __init__(self):
        super().__init__()
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 4)
        )
    
    def forward(self, input_values):
        outputs = self.hubert(input_values)
        pooled = outputs.last_hidden_state. mean(dim=1)
        return self.head(pooled)

print("Loading models...")
# WavLM
wavlm_predictor = initialize_predictor()

# HuBERT
hubert_model = HuBERTAestheticsPredictor()
checkpoint = torch.load('models/hubert_aesthetics_epoch10.pt')
hubert_model.load_state_dict(checkpoint['model_state_dict'])
hubert_model.eval()

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")

# Test on random audio
test_audio = "audio/1-100032-A-0.wav"

print(f"\nTesting on:  {test_audio}\n")

# WavLM prediction
wavlm_preds = wavlm_predictor. forward([{"path": test_audio}])
print("WavLM (Teacher):")
print(f"  CE: {wavlm_preds[0]['CE']:.2f}")
print(f"  CU: {wavlm_preds[0]['CU']:.2f}")
print(f"  PC: {wavlm_preds[0]['PC']:.2f}")
print(f"  PQ: {wavlm_preds[0]['PQ']:.2f}")

# HuBERT prediction
waveform, sr = torchaudio.load(test_audio)
if sr != 16000:
    resampler = torchaudio. transforms.Resample(sr, 16000)
    waveform = resampler(waveform)
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

inputs = feature_extractor(
    waveform.squeeze().numpy(),
    sampling_rate=16000,
    return_tensors="pt"
)

with torch.no_grad():
    hubert_output = hubert_model(inputs['input_values'])

print("\nHuBERT (Student):")
print(f"  CE: {hubert_output[0][0]. item():.2f}")
print(f"  CU: {hubert_output[0][1].item():.2f}")
print(f"  PC: {hubert_output[0][2].item():.2f}")
print(f"  PQ:  {hubert_output[0][3].item():.2f}")

print("\n✅ Comparison complete!")
