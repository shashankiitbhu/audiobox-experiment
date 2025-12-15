# debug_inference.py
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
torch.set_default_device('cpu')

import torchaudio
torchaudio.set_audio_backend("soundfile")

from audiobox_aesthetics.infer import initialize_predictor
import numpy as np

print("="*70)
print("DEBUGGING THE INFERENCE PIPELINE")
print("="*70)

# Load model
print("\n[1] Loading model...")
predictor = initialize_predictor()
predictor.device = torch.device('cpu')
predictor.model = predictor.model.to('cpu')
print(f"    Device: {predictor.device}")
print(f"    Sample rate: {predictor.sample_rate} Hz")

# Load audio manually to see what it looks like
print("\n[2] Loading audio file...")
wav, sr = torchaudio.load("test_aud.mp3")
print(f"    Original shape: {wav.shape}")
print(f"    Original sample rate: {sr} Hz")
print(f"    Duration: {wav.shape[1] / sr:.2f} seconds")
print(f"    Channels: {wav.shape[0]} ({'stereo' if wav.shape[0] == 2 else 'mono'})")

# Show raw values
print(f"\n    First 10 samples: {wav[0, :10].numpy()}")
print(f"    Min value: {wav.min():.4f}")
print(f"    Max value: {wav.max():.4f}")
print(f"    Mean value:  {wav.mean():.4f}")

# Resample to 16kHz
print("\n[3] Resampling to 16kHz...")
wav_resampled = torchaudio. functional.resample(wav, sr, 16000)
print(f"    New shape:  {wav_resampled.shape}")
print(f"    New duration: {wav_resampled. shape[1] / 16000:.2f} seconds")

# Convert to mono
print("\n[4] Converting to mono...")
if wav_resampled.shape[0] > 1:
    wav_mono = wav_resampled.mean(dim=0, keepdim=True)
    print(f"    Averaged {wav_resampled.shape[0]} channels")
else:
    wav_mono = wav_resampled
    print(f"    Already mono")
print(f"    Final shape: {wav_mono.shape}")

# Pad/trim to exactly 10 seconds
print("\n[5] Padding/trimming to 10 seconds...")
target_length = 16000 * 10  # 10 seconds at 16kHz = 160,000 samples
current_length = wav_mono.shape[1]
print(f"    Current length: {current_length} samples ({current_length/16000:.2f}s)")
print(f"    Target length: {target_length} samples (10.00s)")

if current_length < target_length:
    padding_needed = target_length - current_length
    wav_mono = torch.nn.functional.pad(wav_mono, (0, padding_needed))
    print(f"    ✓ Padded with {padding_needed} zeros")
else:
    wav_mono = wav_mono[:, :target_length]
    print(f"    ✓ Trimmed to {target_length} samples")

print(f"    Final shape: {wav_mono.shape}")

# Create batch for model
print("\n[6] Creating batch...")
batch = {
    "wav": wav_mono. unsqueeze(0),  # Add batch dimension [1, 1, 160000]
"mask": torch.ones(1, 1, target_length, dtype=torch.bool)  # Boolean tensor ✓
}
print(f"    Batch wav shape: {batch['wav'].shape}")
print(f"    Batch mask shape: {batch['mask'].shape}")

# Run through model
print("\n[7] Running through model...")
print("    This will take 5-15 seconds on CPU...")

with torch.inference_mode():
    preds = predictor.model(batch)

print("    ✓ Model forward pass complete!")

# Show raw predictions (normalized)
print("\n[8] Raw predictions (normalized):")
for axis in ["CE", "CU", "PC", "PQ"]:
    raw_value = preds[axis].item()
    print(f"    {axis}: {raw_value:.4f}")

# Denormalize
print("\n[9] Denormalizing to 1-10 scale:")
for axis in ["CE", "CU", "PC", "PQ"]: 
    raw_value = preds[axis].item()
    mean = predictor.model.target_transform[axis]["mean"]
    std = predictor.model.target_transform[axis]["std"]
    
    final_value = raw_value * std + mean
    
    print(f"    {axis}:")
    print(f"        Normalized: {raw_value:.4f}")
    print(f"        Mean: {mean:.4f}, Std: {std:.4f}")
    print(f"        Final:  {final_value:.2f}/10")

print("\n" + "="*70)
print("DEBUG COMPLETE!")
print("="*70)