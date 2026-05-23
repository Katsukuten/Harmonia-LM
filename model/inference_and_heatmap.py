"""
Inference & Sampling Engine (Generation Pipeline)

What is this and why does it matter?
This script takes the trained model weights and uses them to autoregressively generate new symbolic music through
Hugging Face's GenerationConfig.
It includes advanced decoding parameters (Temperature, Top-P, Top-K, Repetition Penalty) to shape the 
creative output of the model. 

Attention Heatmap:
An optional toggle allows the generation of an Attention Heatmap. This is a crucial interpretability 
tool. By performing a final forward pass and extracting the attention weights of the last transformer layer, 
we can visually map out which prior tokens the model "looked at" to make its final decisions. To prevent 
memory overload and unreadable plots, the heatmap isolates the structural relationships of a specific window.
"""

import torch
from pathlib import Path
from miditok import TSD, TokenizerConfig
from transformers import GenerationConfig
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
from training import Qwen3MusicModel 
import gc
import torch.serialization
torch.serialization.add_safe_globals([TSD, TokenizerConfig]) # Add Miditok to torch's whitelist to allow operations

def cleanup_gpu(model):
    """
    Force the explicit release of GPU resources to prevent VRAM accumulation 
    in interactive sessions.
    """
    del model
    torch.cuda.empty_cache()
    gc.collect()
    print("GPU Memory cleared.")

def main():
    print("--- Inference Engine (sampling & visualization) ---")

    # Path management
    CURRENT_DIR = Path(__file__).resolve().parent
    DATABASE = CURRENT_DIR.parent / "database"
    CHECKPOINT_DIR = CURRENT_DIR / "Qwen3" / "checkpoints"
    
    OUTPUT_GEN_DIR = CURRENT_DIR / "generated_music"
    OUTPUT_GEN_DIR.mkdir(parents=True, exist_ok=True)

    # Sampling configuration
    primer_midi_path = None # Set to None for ex nihilo inference
    primer_len = 1024 if primer_midi_path else 0

    # Heatmap switch
    GENERATE_HEATMAP = True
    HEATMAP_WINDOW_SIZE = 100
    
    MAX_TOKENS = 4096
    tokens_to_generate = MAX_TOKENS - primer_len
    
    # --- Sampling Theory ---
    # Temperature: Entropy control. 
    # < 1.0: The model takes few risks (very strict, perfect for Bach).
    # > 1.0: The model tries wilder connections (jazzy, but risks dissonance).
    temperature = 0.95

    # Top-P (Nucleus Sampling): Tails clipping.
    # 0.95 means we eliminate the bottom 5% of probabilities (the obvious "wrong notes").
    top_p = 0.95

    # Top-K: Strict barrier.
    # If k=15, the model can only choose among the 15 most likely tokens at each step.
    # Highly useful in music to force the model to stay within a scale/chord. Set to 0 to disable.
    k = 0

    # Repetition Penalty: The loop breaker.
    # If the model gets stuck playing a motif, increase this (e.g., 1.1 or 1.2).
    # If it never repeats a motif (which sounds unmusical), leave it at 1.0.
    repetition_penalty = 1.0

    # Loading (Tokenizer & Model Weights)
    tokenizer_path = DATABASE / "tokenizer.json"
    if not tokenizer_path.exists():
        print(f"Error: Tokenizer not found at -> {tokenizer_path}")
        return

    tokenizer = TSD(params=tokenizer_path)
    print(f"Tokenizer loaded (Vocab size: {tokenizer.vocab_size} tokens)")

    checkpoints = list(CHECKPOINT_DIR.glob("*.ckpt"))
    if not checkpoints:
        print(f"No checkpoint found in {CHECKPOINT_DIR}")
        return
    best_ckpt = max(checkpoints, key=lambda x: x.stat().st_mtime)
    
    print(f"Loading weights from: {best_ckpt.name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Bypassing Pytorch's 2.6 security patch (safe since the checkpoint is made here)
    _original_load = torch.load
    def _trusted_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_load(*args, **kwargs)
        
    torch.load = _trusted_load
        
    try:
        model = Qwen3MusicModel.load_from_checkpoint(
            best_ckpt, tokenizer=tokenizer, tokenizer_vocab_size=tokenizer.vocab_size
        )
    finally:
        torch.load = _original_load
    model.eval().to(device)
    print(f"Model ready on {device}")

    try:
        # Primer preparation
        bos_token_id = tokenizer["BOS_None"] if "BOS_None" in tokenizer.vocab else tokenizer["PAD_None"]

        if primer_midi_path and Path(primer_midi_path).exists():
            primer_name = Path(primer_midi_path).stem[:15]
            print(f"Loading primer: {Path(primer_midi_path).name}")
            
            primer_tokens = tokenizer(primer_midi_path)
            ids = primer_tokens[0].ids if hasattr(primer_tokens[0], 'ids') else primer_tokens[0]
            
            if len(ids) > primer_len:
                ids = ids[:primer_len]
            
            if ids[0] != bos_token_id:
                ids.insert(0, bos_token_id)
                
            input_ids = torch.tensor([ids], dtype=torch.long).to(device)
            print(f"Primer loaded ({len(ids)} tokens).")
        else:
            primer_name = "None"
            print("No primer detected, generating from BOS.")
            input_ids = torch.tensor([[bos_token_id]], dtype=torch.long).to(device)

        # Generation engine
        gen_config = GenerationConfig(
            max_new_tokens=tokens_to_generate,
            min_new_tokens=tokens_to_generate,
            do_sample=True,
            temperature=temperature,
            top_k=k,
            top_p=top_p,
            eos_token_id=tokenizer["EOS_None"] if "EOS_None" in tokenizer.vocab else tokenizer["PAD_None"],
            pad_token_id=tokenizer["PAD_None"],
            repetition_penalty=repetition_penalty
        )

        print(f"Launching generation... (temp={temperature}, p={top_p}, k={k}, rep={repetition_penalty})")

        with torch.no_grad():
            try:
                outputs = model.model.generate(
                    inputs=input_ids,
                    generation_config=gen_config,
                    use_cache=True,
                )
            except RuntimeError as e:
                print(f"Error during generation: {e}")
                return # Python met ce return en pause, exécute le finally, puis quitte le script.

        generated_ids = outputs[0].tolist()
        
        # Standardized Nomenclature
        prefix = f"primer-{primer_name}_" if primer_midi_path else "scratch_"
        filename_stem = f"{prefix}len-{len(generated_ids)}_temp-{temperature}_topP-{top_p}_topK-{k}_rep-{repetition_penalty}"

        # Optional: Attention heatmap visualization
        if GENERATE_HEATMAP:
            print("Computing Attention Heatmap ...")
            try:
                with torch.no_grad():
                    attn_outputs = model.model(torch.tensor([generated_ids]).to(device), output_attentions=True)
                
                last_layer_attention = attn_outputs.attentions[-1]
                avg_attention = last_layer_attention[0].mean(dim=0).cpu().numpy()
                
                seq_length = avg_attention.shape[0]
                window = min(HEATMAP_WINDOW_SIZE, seq_length)
                cropped_attention = avg_attention[-window:, -window:]
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(cropped_attention, cmap='magma', vmin=0, vmax=float(cropped_attention.max()))
                plt.title(f"Causal Attention Heatmap (Last Layer, Last {window} tokens)")
                plt.xlabel("Key Tokens (Look-back)")
                plt.ylabel("Query Tokens (Current position)")
                
                heatmap_path = OUTPUT_GEN_DIR / f"{filename_stem}_heatmap.png"
                plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Heatmap successfully saved to: {heatmap_path.name}")
            except Exception as e:
                print(f"Could not generate heatmap: {e}")

        # Decoding & strict persistence
        print("Decoding tokens back into MIDI ...")
        midi_path = OUTPUT_GEN_DIR / f"{filename_stem}.mid"

        try:
            midi_obj = tokenizer.decode([deepcopy(generated_ids)])
            midi_bytes = midi_obj.dumps_midi()
            
            with open(midi_path, "wb") as f:
                f.write(midi_bytes)
                
            print(f"MIDI File generated and saved here: {midi_path.name}")
            
        except Exception as e:
            print(f"Error during MIDI conversion: {e}")
            backup_path = OUTPUT_GEN_DIR / f"{filename_stem}.txt"
            with open(backup_path, "w") as f:
                f.write(str(generated_ids))
            print(f"Raw IDs were emergency-saved in {backup_path.name}")

    finally:
        cleanup_gpu(model)
if __name__ == "__main__":
    main()
