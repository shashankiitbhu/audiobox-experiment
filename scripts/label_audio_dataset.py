# scripts/label_audio_dataset.py
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
torch.set_default_device('cpu')

from audiobox_aesthetics.infer import initialize_predictor
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import time

print("=" * 70)
print("LABELING AUDIO WITH WAVLM (TEACHER MODEL)")
print("=" * 70)

AUDIO_DIR = Path("audio")
OUTPUT_CSV = "data/audio_labels.csv"

Path("data").mkdir(exist_ok=True)

print(f"\n[1] Finding audio files...")
audio_files = sorted(AUDIO_DIR.glob("*. wav"))
print(f"    Found {len(audio_files)} files")

print(f"\n[2] Loading WavLM model...")
predictor = initialize_predictor()
predictor.device = torch.device('cpu')
predictor.model = predictor.model.to('cpu')
print("    ✅ Model loaded")

print(f"\n[3] Labeling {len(audio_files)} files...")
print("    (This will take ~1-2 hours on CPU)\n")

results = []
start_time = time.time()

for audio_path in tqdm(audio_files, desc="Labeling"):
    try:
        preds = predictor.forward([{"path": str(audio_path)}])
        
        results.append({
            'filename': audio_path.name,
            'filepath': str(audio_path),
            'CE':  preds[0]['CE'],
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

print(f"    ✅ Saved {len(results)} labels")
print(f"    Time:  {elapsed/60:.1f} minutes")

print(f"\n[5] Label Statistics:")
for axis in ['CE', 'CU', 'PC', 'PQ']:
    vals = df[axis].values
    print(f"    {axis}:  mean={vals.mean():.2f}, std={vals.std():.2f}, min={vals.min():.2f}, max={vals.max():.2f}")

print(f"\n[6] Sample labels:")
print(df. head(10).to_string())

print("\n" + "=" * 70)
print("✅ LABELING COMPLETE!")
print("=" * 70)
print("\nNext steps:")
print("1. git add data/audio_labels.csv")
print("2. git commit -m 'Add WavLM labels'")
print("3. git push")
print("4. Clone on cloud GPU and train HuBERT")