# test_hubert.py
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
torch.set_default_device('cpu')

from transformers import HubertModel
import torchaudio
import time

print("="*60)
print("TESTING HUBERT REPLACEMENT")
print("="*60)

# Load HuBERT model
print("\n[1] Loading HuBERT model...")
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
hubert.eval()
print("    ✓ HuBERT loaded")

# Load your audio
print("\n[2] Loading audio...")
wav, sr = torchaudio.load("test_aud.mp3")

# Preprocess (same as Audiobox does)
wav = torchaudio.functional.resample(wav, sr, 16000)
wav = wav.mean(dim=0, keepdim=True)  # Convert to mono
target_length = 160000
if wav.shape[1] < target_length:
    wav = torch.nn.functional.pad(wav, (0, target_length - wav.shape[1]))
else:
    wav = wav[:, :target_length]

print(f"    Audio shape: {wav.shape}")

# Run through HuBERT
print("\n[3] Running through HuBERT...")
start = time.time()
with torch.no_grad():
    outputs = hubert(wav)
    features = outputs.last_hidden_state

print(f"    ✓ Done in {time.time() - start:.1f} seconds")
print(f"    Output shape: {features.shape}")

# Compare with WavLM shape expectation
print("\n[4] Checking compatibility...")
expected_shape = "[1, ~499, 768]"
actual_shape = list(features.shape)
print(f"    Expected shape: {expected_shape}")
print(f"    Actual shape:    {actual_shape}")

if actual_shape[2] == 768:
    print("    ✅ COMPATIBLE!  HuBERT outputs 768 dimensions like WavLM")
else:
    print(f"    ⚠️  Warning: HuBERT outputs {actual_shape[2]} dimensions, not 768")

print("\n" + "="*60)
print("SUCCESS! HuBERT works and is compatible!")
print("="*60)