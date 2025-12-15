import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
from audiobox_aesthetics.infer import initialize_predictor
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import time

print("=" * 70)
print("LABELING AUDIO WITH WAVLM (TEACHER MODEL)")
print("=" * 70)

AUDIO_DIR = Path('/home/pixaverse.admin/shashank/audiobox-experiment/audio')
OUTPUT_CSV = Path('/home/pixaverse.admin/shashank/audiobox-experiment/data/audio_labels.csv')

OUTPUT_CSV.parent.mkdir(exist_ok=True)

print(f"\n[1] Finding audio files...")
audio_files = sorted(AUDIO_DIR.glob('*.wav'))
print(f"    Found {len(audio_files)} files")

assert len(audio_files) > 0, "No files found!"

print(f"\n[2] Loading WavLM model...")
predictor = initialize_predictor()
print("    ✅ Model loaded")

print(f"\n[3] Labeling {len(audio_files)} files.. .\n")

results = []
start_time = time.time()

for audio_path in tqdm(audio_files, desc="Labeling"):
    try:
        preds = predictor.forward([{"path": str(audio_path)}])
        
        results.append({
            'filename': audio_path.name,
            'filepath': str(audio_path),
            'CE': preds[0]['CE'],
            'CU': preds[0]['CU'],
            'PC': preds[0]['PC'],
            'PQ': preds[0]['PQ']
        })
    except Exception as e:
        print(f"\n⚠️ Error on {audio_path.name}: {e}")

elapsed = time.time() - start_time

print(f"\n[4] Saving to {OUTPUT_CSV}...")
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)

print(f"    ✅ Saved {len(results)} labels in {elapsed/60:.1f} minutes")

print(f"\n[5] Label Statistics:")
for axis in ['CE', 'CU', 'PC', 'PQ']:
    vals = df[axis].values
    print(f"    {axis}: mean={vals.mean():.2f}, std={vals.std():.2f}, range=[{vals.min():.2f}, {vals.max():.2f}]")

print("\n" + "=" * 70)
print("✅ DONE!  Labels saved to:", OUTPUT_CSV)
print("=" * 70)
