
#!/usr/bin/env python3
import torch
import torchaudio
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from audiobox_aesthetics. infer import initialize_predictor

# Load HuBERT
class HuBERTAestheticsPredictor(nn.Module):
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

print("=" * 70)
print("EVALUATING WAVLM vs HUBERT")
print("=" * 70)

print("\n[1] Loading models...")
wavlm_predictor = initialize_predictor()
hubert_model = HuBERTAestheticsPredictor()
checkpoint = torch.load('models/hubert_aesthetics_epoch10.pt')
hubert_model.load_state_dict(checkpoint['model_state_dict'])
hubert_model.eval()
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
print("    ✅ Models loaded")

print("\n[2] Loading ground truth labels...")
df = pd.read_csv('data/audio_labels.csv')
# Use last 200 samples as test set (not seen during detailed analysis)
test_df = df.tail(200)
print(f"    ✅ Testing on {len(test_df)} samples")

print("\n[3] Running inference.. .\n")

hubert_predictions = []

for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="HuBERT inference"):
    # Load audio
    waveform, sr = torchaudio.load(row['filepath'])
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Predict
    inputs = feature_extractor(
        waveform.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        output = hubert_model(inputs['input_values'])
    
    hubert_predictions.append({
        'filename': row['filename'],
        'CE_hubert': output[0][0]. item(),
        'CU_hubert': output[0][1].item(),
        'PC_hubert': output[0][2].item(),
        'PQ_hubert': output[0][3].item(),
        'CE_wavlm': row['CE'],
        'CU_wavlm': row['CU'],
        'PC_wavlm':  row['PC'],
        'PQ_wavlm': row['PQ']
    })

results_df = pd.DataFrame(hubert_predictions)

print("\n[4] Computing metrics.. .\n")

print("=" * 70)
print("📊 MEAN ABSOLUTE ERROR (MAE)")
print("=" * 70)

for axis in ['CE', 'CU', 'PC', 'PQ']:
    mae = np.abs(results_df[f'{axis}_hubert'] - results_df[f'{axis}_wavlm']).mean()
    print(f"{axis}:   {mae:.3f}")

print("\n" + "=" * 70)
print("📈 CORRELATION")
print("=" * 70)

for axis in ['CE', 'CU', 'PC', 'PQ']:
    corr = np.corrcoef(results_df[f'{axis}_hubert'], results_df[f'{axis}_wavlm'])[0, 1]
    print(f"{axis}:  {corr:.3f}")

print("\n" + "=" * 70)
print("✅ EVALUATION COMPLETE")
print("=" * 70)

# Save results
results_df.to_csv('data/evaluation_results.csv', index=False)
print(f"\n💾 Results saved to:  data/evaluation_results.csv")
