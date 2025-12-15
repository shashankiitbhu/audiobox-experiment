# audiobox_hubert.py
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
torch.set_default_device('cpu')

import torchaudio
from transformers import HubertModel
from audiobox_aesthetics. infer import initialize_predictor
import time

print("="*60)
print("AUDIOBOX WITH HUBERT")
print("="*60)

# Load the original predictor (to get the trained heads)
print("\n[1] Loading original Audiobox model...")
predictor = initialize_predictor()
predictor.device = torch.device('cpu')
predictor.model = predictor.model.to('cpu')

# Replace WavLM with HuBERT
print("\n[2] Replacing WavLM with HuBERT...")
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
hubert.eval()
hubert. to('cpu')

# Monkey-patch the encoder
predictor.model.wavlm_model = hubert
print("    ✓ WavLM replaced with HuBERT!")

# Modify the forward pass to use HuBERT's output format
original_forward = predictor.model. forward

def hubert_forward(self, batch):
    """Modified forward pass for HuBERT"""
    wav = batch["wav"]. squeeze(1)
    
    # Run through HuBERT
    with torch.set_grad_enabled(self.training):
        outputs = self.wavlm_model(wav)  # Now calling HuBERT
        last_hidden = outputs.last_hidden_state  # [batch, time, 768]
        
    # Transpose to match WavLM format:  [time, batch, 768]
    last_hidden = last_hidden.transpose(0, 1)
    
    # Create fake "all layers" output (HuBERT only gives last layer)
    # We'll just repeat the last layer 13 times
    all_outputs = [(last_hidden, None)] * 13
    
    # Stack layers
    all_outputs_stacked = torch.stack([x[0] for x in all_outputs], dim=-1)
    # Shape: [time, batch, 768, 13]
    
    preds = {}
    
    # For each axis
    for name in ["CE", "CU", "PC", "PQ"]:
        if self.use_weighted_layer_sum:
            norm_weights = torch.nn.functional.softmax(
                self.layer_weights[name], dim=-1
            )
            audio_embed = torch.einsum("tbcl,l->btc", all_outputs_stacked, norm_weights)
        else:
            audio_embed = last_hidden. transpose(0, 1)
        
        # Temporal pooling
        audio_embed = audio_embed.mean(dim=1)
        
        # Normalize
        if self.normalize_embed:
            audio_embed = torch.nn.functional.normalize(audio_embed, dim=-1)
        
        # MLP prediction
        preds[name] = self.proj_layer[name](audio_embed).squeeze(-1)
    
    return preds

# Replace the forward method
import types
predictor.model.forward = types.MethodType(hubert_forward, predictor.model)

print("\n[3] Testing on audio file...")
print("    Processing audio...")

start = time.time()
results = predictor.forward([{"path": "test_aud.mp3"}])
duration = time.time() - start

print(f"    ✓ Completed in {duration:.1f} seconds")

# Print results
print("\n[4] RESULTS (with HuBERT):")
print("="*60)

axis_names = {
    "CE": "Content Enjoyment    ",
    "CU": "Content Usefulness  ",
    "PC": "Production Complexity",
    "PQ": "Production Quality   "
}

for axis, score in results[0].items():
    bar_length = int(score)
    bar = "█" * bar_length + "░" * (10 - bar_length)
    print(f"{axis_names[axis]} ({axis}): {score:.2f}/10")
    print(f"  [{bar}]")

print("="*60)
print("SUCCESS! Audiobox now uses HuBERT instead of WavLM!")
print("="*60)