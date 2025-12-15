# download_freesound.py
import requests
import os
from tqdm import tqdm

print("Downloading from Freesound...")

os.makedirs("audio", exist_ok=True)

# Direct URLs from Freesound (creative commons)
# These are direct . wav/. mp3 links that work
base_url = "https://freesound.org/data/previews/"

# Sample IDs (you can add more)
sound_ids = list(range(1, 1001))

saved = 0

for i in tqdm(sound_ids, desc="Downloading"):
    # Try both mp3 and wav
    for ext in ['mp3', 'wav']: 
        try:
            # Construct URL (Freesound preview format)
            folder = str(i // 1000).zfill(3)
            url = f"{base_url}{folder}/{i}_preview.{ext}"
            
            response = requests.get(url, timeout=5)
            
            if response. status_code == 200 and len(response.content) > 1000:
                output = f"audio/{i:05d}.{ext}"
                with open(output, 'wb') as f:
                    f. write(response.content)
                saved += 1
                break
        except: 
            continue
    
    if saved >= 100:  # Get at least 100 files
        break

print(f"\n✅ Downloaded {saved} files")