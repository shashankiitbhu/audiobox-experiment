# test_model.py
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
# Force CPU usage
torch.set_default_device('cpu')

import torchaudio
torchaudio.set_audio_backend("soundfile")

from audiobox_aesthetics.infer import initialize_predictor
import time

print("="*60)
print("AUDIOBOX AESTHETICS - FIRST RUN")
print("="*60)

# Initialize the model (downloads checkpoint on first run)
print("\n[1/3] Loading model...")
print("      (First run will download ~400MB checkpoint)")
print("      This may take 2-5 minutes depending on internet speed...")

start_time = time.time()
predictor = initialize_predictor()

# Force model to CPU
predictor.device = torch.device('cpu')
predictor.model = predictor. model.to('cpu')

load_time = time.time() - start_time

print(f"      ✓ Model loaded in {load_time:.1f} seconds")

# Predict scores for your audio
print("\n[2/3] Running prediction...")
print("      Processing audio file...")

pred_start = time.time()
results = predictor.forward([
    {"path": "test_aud.mp3"}  # Change to test_audio.mp3 if you used MP3
])
pred_time = time.time() - pred_start

print(f"      ✓ Prediction completed in {pred_time:.1f} seconds")

# Print results
print("\n[3/3] RESULTS:")
print("="*60)

axis_names = {
    "CE": "Content Enjoyment    ",
    "CU": "Content Usefulness  ",
    "PC": "Production Complexity",
    "PQ": "Production Quality   "
}

axis_descriptions = {
    "CE": "How enjoyable/entertaining is the content?",
    "CU": "How useful/informative is the content?",
    "PC": "How complex is the production?  (simple voice vs full production)",
    "PQ": "What is the technical quality?   (noise, clarity, etc.)"
}

for axis, score in results[0].items():
    bar_length = int(score)
    bar = "█" * bar_length + "░" * (10 - bar_length)
    
    print(f"\n{axis_names[axis]} ({axis}): {score:.2f}/10")
    print(f"  [{bar}]")
    print(f"  {axis_descriptions[axis]}")

print("\n" + "="*60)
print("SUCCESS! Your first model run is complete!   🎉")
print("="*60)