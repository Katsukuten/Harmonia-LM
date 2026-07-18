"""
Initial Weight Matrix Conditioning Analysis Callback

What is this and why does it matter?
This callback analyzes the condition number (Kappa) of our model's weight matrices at t=0, right before
training starts. The condition number is computed as the ratio of the maximum singular value to the
minimum singular value (Kappa = sigma_max / sigma_min).
Concretely, it tells us how much a linear transformation distorts its latent space. An optimal condition
number is 1.0, meaning the matrix acts as a perfect isometry—preserving distances and angles. This is crucial
at initialization to ensure smooth, stable gradient propagation and avoid vanishing/exploding gradients.

WARNING / MODEL DEPENDENCY:
This file is NOT fully plug-and-play. It was meticulously tailored for the Qwen3 architecture. I had to dig into
the model's source code to map out the exact internal names of each weight matrix. If you swap Qwen3 for another
LLM (like Llama or Mistral), the matrix names inside 'category_patterns' WILL change, and you will have to update
them manually. While this requires custom tweaks for every new architecture, it is the absolute best way to ensure
maximum monitoring and granular control over what is happening under the hood.

Mathematical Insight on Embeddings vs. Other Layers:
Don't waste time looking at a global network average—it doesn't mean much here. The absolute priority is the
Embedding matrix. Thanks to the custom orthogonal initialization, the embedding weights will achieve a perfect
condition number of 1.0, freezing a clean geometric structure from step zero, even though the matrix is non-square
(e.g., 666 by 512), and most importantly : an unbiased starting point, each token being unrelated to one another.
On the other hand, expect the other projection matrices (like Attention or MLP blocks) to show much higher condition
numbers. This is completely normal and intended: it's the direct mathematical consequence of the variance scaling
(like GPT-2 scaling) applied to deep residual paths to prevent signal collapse across multiple layers.
"""

import torch
import pytorch_lightning as pl
import numpy as np
from prettytable import PrettyTable
from pathlib import Path


class InitialConditionAnalysis(pl.Callback):
    """
    Analyzes the conditioning of weight matrices at t=0.
    A condition number close to 1 indicates a near-isometric matrix,
    which is ideal for faster and more stable convergence.
    """

    def __init__(self, output_file="conditioning_report.txt"):
        super().__init__()
        self.output_file = Path(output_file)

        # Core layer mapping — specific to Qwen3's source code nomenclature.
        # Change these strings if you are switching to another model architecture.
        self.category_patterns = {
            "Embeddings": ["embed_tokens"],
            "Attn_Q": ["self_attn.q_proj"],
            "Attn_K": ["self_attn.k_proj"],
            "Attn_V": ["self_attn.v_proj"],
            "Attn_Output": ["self_attn.o_proj"],
            "MLP_Gate": ["mlp.gate_proj"],
            "MLP_Up": ["mlp.up_proj"],
            "MLP_Down": ["mlp.down_proj"],
            "LM_Head": ["lm_head"],
        }

    def _compute_cond(self, tensor):
        if tensor is None or tensor.dim() < 2:
            return None

        # Keeping computation on CUDA since SVD is heavily accelerated by my GPU setup.
        # Enforcing float32 because SVD algorithms are notoriously unstable in half/bf16 precision.
        with torch.no_grad():
            matrix = tensor.detach().float()
            try:
                # Singular value decomposition via CuSOLVER
                s = torch.linalg.svdvals(matrix)
                # Safety epsilon against singular matrices to prevent division by zero
                eps = 1e-9
                cond = s[0] / (s[-1] + eps)
                return cond.item()
            except RuntimeError:
                return None

    def on_train_start(self, trainer, pl_module):
        print("--- Initial Conditioning Analysis ---")

        table = PrettyTable()
        table.field_names = ["Layer Type", "Mean Kappa", "Min", "Max"]

        layer_stats = {cat: [] for cat in self.category_patterns}
        layer_stats["Other"] = []

        # Scanning the model parameters
        for name, param in pl_module.model.named_parameters():
            if param.dim() < 2:
                continue

            category = next(
                (
                    cat
                    for cat, patterns in self.category_patterns.items()
                    if any(p in name for p in patterns)
                ),
                "Other",
            )

            cond = self._compute_cond(param)
            if cond is not None:
                layer_stats[category].append(cond)

        global_avg = []
        for cat, values in layer_stats.items():
            if values:
                avg, mn, mx = np.mean(values), np.min(values), np.max(values)
                global_avg.extend(values)
                table.add_row([cat, f"{avg:.2f}", f"{mn:.2f}", f"{mx:.2f}"])

                if trainer.logger:
                    trainer.logger.experiment.add_scalar(f"Init_Cond/{cat}", avg, 0)

        print(table)
        global_stat_str = ""
        if global_avg:
            global_stat_str = f"\nGlobal Mean Conditioning: {np.mean(global_avg):.2f}\n"
            print(global_stat_str)
        print("=" * 65 + "\n")

        # Report file persistence
        try:
            # Making sure the parent directory exists before writing
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_file, "w", encoding="utf-8") as f:
                f.write("--- Initial Conditioning Report ---\n")
                f.write(table.get_string() + "\n")
                f.write(global_stat_str)
                f.write("=" * 65 + "\n")
            print(f"Report successfully saved to: {self.output_file.absolute()}")
        except Exception as e:
            print(f"Error while saving the report: {e}")


if __name__ == "__main__":
    pass
