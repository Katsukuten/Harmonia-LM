"""
Latent Space Topological Analysis (UMAP Projection)
Read UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction, McInnes et al., 2018

This script extracts the embedding matrix from the model to visualize how musical concepts 
(Pitch, Velocity, Duration, TimeShift) are organized in the latent space.

Configuration:
- INITIAL_MODE: Set to True to analyze weights at t=0 (post-initialization), 
                Set to False to analyze the model after training (loads last checkpoint).
- N_DIM: Set to 2 for 2D projection, 3 for 3D.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
from pathlib import Path
import pytorch_lightning as pl
from miditok import TSD
from training import Qwen3MusicModel
import gc 

# --- Configuration ---
INITIAL_MODE = False  # True: Cold Init (Unbiased), False: Trained (Checkpoint)
N_DIM = 3            # 2 or 3

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
    print(f"--- LATENT SPACE ANALYSIS (UMAP {N_DIM}D | INIT={INITIAL_MODE}) ---")
    
    # Path Configuration
    CURRENT_DIR = Path(__file__).resolve().parent
    DATABASE = CURRENT_DIR.parent / "database"
    CHECKPOINT_DIR = CURRENT_DIR / "Qwen3" / "checkpoints"
    
    # Initialization
    tokenizer = TSD(params=DATABASE / "tokenizer.json")
    
    if INITIAL_MODE:
        print("Initializing model from scratch (t=0)...")
        model = Qwen3MusicModel(
            tokenizer=tokenizer,
            tokenizer_vocab_size=tokenizer.vocab_size,
            batch_size=8,
            learning_rate=6e-4,
            epochs=1,
            dataset_size=1,
            resume=False
        )
    else:
        checkpoints = list(CHECKPOINT_DIR.glob("*.ckpt"))
        if not checkpoints:
            print("Error: No checkpoint found.")
            return
        best_ckpt = max(checkpoints, key=lambda x: x.stat().st_mtime)
        print(f"Loading weights from: {best_ckpt.name}...")
        model = Qwen3MusicModel.load_from_checkpoint(
            best_ckpt, tokenizer=tokenizer, tokenizer_vocab_size=tokenizer.vocab_size
        )
    
    try:
        # Embedding Extraction
        embeddings = model.model.model.embed_tokens.weight.detach().cpu().numpy()
        
        # Categorization
        vocab = tokenizer.vocab
        ids, labels, colors = [], [], []
        color_map = {"Pitch": "blue", "Velocity": "green", "Duration": "red", "TimeShift": "orange", "Special": "gray"}
        
        for token_str, token_id in vocab.items():
            ids.append(token_id)
            if token_str.startswith("Pitch"): labels.append("Pitch"); colors.append(color_map["Pitch"])
            elif token_str.startswith("Velocity"): labels.append("Velocity"); colors.append(color_map["Velocity"])
            elif token_str.startswith("Duration"): labels.append("Duration"); colors.append(color_map["Duration"])
            elif token_str.startswith("TimeShift"): labels.append("TimeShift"); colors.append(color_map["TimeShift"])
            else: labels.append("Special"); colors.append(color_map["Special"])

        # UMAP Projection
        print(f"Computing UMAP {N_DIM}D projection...")
        reducer = umap.UMAP(
            n_components=N_DIM, 
            n_neighbors=15, 
            min_dist=0.1,
            random_state=42,
            metric='cosine'
        )
        proj = reducer.fit_transform(embeddings[ids])

        # Visualization
        fig = plt.figure(figsize=(12, 10 if N_DIM == 3 else 8))
        ax = fig.add_subplot(111, projection='3d' if N_DIM == 3 else None)
        
        for label in set(labels):
            idx = [i for i, l in enumerate(labels) if l == label]
            if N_DIM == 2:
                ax.scatter(proj[idx, 0], proj[idx, 1], c=[colors[i] for i in idx], label=label, alpha=0.8, edgecolors='w', linewidth=0.5)
            else:
                ax.scatter(proj[idx, 0], proj[idx, 1], proj[idx, 2], c=[colors[i] for i in idx], label=label, alpha=0.8, edgecolors='w', linewidth=0.5, s=40)
                
        ax.set_title(f"Musical Latent Space (UMAP {N_DIM}D, Init={INITIAL_MODE})", fontweight='bold')
        
        if N_DIM == 3:
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
        else:
            ax.grid(True, linestyle='--', alpha=0.5)
            
        plt.legend(title="Token Families")
        plt.savefig(CURRENT_DIR / f"umap_{N_DIM}d_initial-{INITIAL_MODE}.png", dpi=300)
        plt.show()

    finally:
        cleanup_gpu(model)

if __name__ == "__main__":
    main()
