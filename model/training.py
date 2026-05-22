"""
LLM Symbolic Music Training Pipeline
This script handles the initialization and full-scale training of a Qwen3 Causal LM on MIDI token sequences.
Architectural & Engineering choices: 
- Hardware-aware PyTorch optimization (BF16, TF32, SDPA/FlashAttention) tailored for Ampere+ architecture (RTX 50 series).
- Custom initializations (Orthogonal embeddings, GPT-2 variance scaling) to ensure stable gradient flow. Totally optional 
  but best for unbiased initialization.
- Memory-bound optimization via custom RAM dataset to bypass standard I/O bottlenecks (might want to revert if insufficient RAM).
- Resumption and Sanity Check functions.

NB: Some warnings might appear at the beginning of this script's execution. I couldn't properly suppress them, 
but they are likely due to library conflicts. The script runs without any fatal errors nonetheless.
"""

import math
import logging
from pathlib import Path
from multiprocessing import freeze_support
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import transformers
from transformers import Qwen3Config, Qwen3ForCausalLM # Change depending on the LLM
from miditok.pytorch_data.collators import DataCollator
from miditok import TSD
# from miditok.pytorch_data import DatasetMIDI (uncomment only if you don't use RAMMidiDataset)
from ram_dataset import RAMMidiDataset
from condition_callback import InitialConditionAnalysis

# ==============================================================================
# 🚨 SANITY CHECK SWITCH 🚨
# Set to True to test convergence (pure overfitting on a single batch)
# Set to False for the REAL full-scale training
# ==============================================================================
SANITY_CHECK = False

transformers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)

# Hardware Optimizations (RTX 50 series / Ampere+ Architecture)
torch.backends.cudnn.benchmark = True  
torch.backends.cuda.matmul.allow_tf32 = True  
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
torch.cuda.empty_cache()

# ==============================================================================
# PATH MANAGEMENT
# You can replace any mention of Qwen3 by any name you want
# ==============================================================================
CURRENT_DIR = Path(__file__).resolve().parent
DATABASE = CURRENT_DIR.parent / "database"
OUTPUT_DIR = CURRENT_DIR / "Qwen3"
ROOT_LOG_DIR = CURRENT_DIR.parent / "logs" / "Qwen3_TSD"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ROOT_LOG_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# GLOBAL HYPERPARAMETERS
# ==============================================================================
SEED = 777 # Set seed for reproducibility
SEQ_LEN = 4096 # Will set the model's context window
BATCH_SIZE, ACC_GRAD = 8, 16 # Effective Batch Size = 128, change depending on your hardware 
STEPS, LR = 60000, 6e-4 # Arbitrary values, tweak only if you know what you're doing
NUM_WORKERS = 0 # Set a reasonable number of your CPU's threads to use, disabled in my case 


# This is the main dish, see Hugging Face's transformers' documentation if you want to change
# the model, usually [model]Config and [model]ForCausalLM.
class Qwen3MusicModel(pl.LightningModule):
    def __init__(self, tokenizer, tokenizer_vocab_size, batch_size,
                 learning_rate, epochs, dataset_size, resume=False):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        
        # Disable dropout during Sanity Check to force pure overfitting
        dropout_prob = 0.0 if SANITY_CHECK else 0.2 
        
        config = Qwen3Config(
            vocab_size=tokenizer_vocab_size,
            hidden_size=512,
            intermediate_size=2048,
            num_hidden_layers=8, 
            num_attention_heads=8,
            num_key_value_heads=8,
            max_position_embeddings=SEQ_LEN,
            tie_word_embeddings=True,
            rope_theta=10000.0,
            attention_dropout=dropout_prob, 
            head_dim=64,
            use_cache=False,
            attn_implementation="sdpa", # Scaled Dot-Product Attention (Native FlashAttention)
            hidden_dropout_prob=dropout_prob 
        )

        # Uncomment ONLY during inference if you plan to create a heatmap (since sdpa never actually materializes or returns the raw attention weight matrix)
        # config._attn_implementation = "eager" 

        self.model = Qwen3ForCausalLM(config)
        self.model.resize_token_embeddings(tokenizer_vocab_size)
        
        if not resume:
            self._apply_custom_initialization(config)
            
    def _apply_custom_initialization(self, config):
        """
        Strict mathematical initialization to force healthy convergence 
        without gradient explosion, optional but recommended.
        Note that if you change models, you'll have to change the aforementioned matrices' names
        by those used in its source code.
        """
        # 1. Orthogonal Projection of Latent Space
        # Preserves L2 norm during initial projections, basically provides an unbiased starting point.
        # If you had to keep one initialization, it'd be this one.
        nn.init.orthogonal_(self.model.model.embed_tokens.weight)

        # 2. GPT-2 Scaling (cf. Language Models are Unsupervised Multitask Learners, Radford et al., 2019)
        # Probability law: Var(Sum of N independent variables) = N * Var(X).
        # To avoid variance explosion across the N residual layers,
        # we divide the standard deviation by sqrt(2 * L).
        base_std = 0.02
        n_residual_layers = 2 * config.num_hidden_layers
        scaled_std = base_std / math.sqrt(n_residual_layers)

        for name, param in self.model.named_parameters():
            if param.dim() >= 2: 
                if "embed_tokens" in name:
                    continue
                # Reduced scaling applies only to residual outputs
                is_residual_output = "o_proj" in name or "down_proj" in name
                std_to_use = scaled_std if is_residual_output else base_std
                nn.init.normal_(param, mean=0, std=std_to_use)

            elif "bias" in name:
                nn.init.constant_(param, 0)

        # RMSNorm Normalization (Unit variance)
        for module in self.model.modules():
            if isinstance(module, (nn.LayerNorm, type(self.model.model.norm))):
                if hasattr(module, 'weight'):
                    nn.init.constant_(module.weight, 1.0)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch):
        outputs = self(**batch)
        val_loss = outputs.loss
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        return val_loss
    
    def on_before_optimizer_step(self, optimizer):
        """
        Robust calculation of the global gradient norm (L2).
        This manual method avoids Lightning's internal key errors.
        """
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.log("grad_norm_total", total_norm, on_step=True, on_epoch=False)

    def configure_optimizers(self):
        """
        I used the "standard" optimizer and schedulers but others might be better for you.
        """
        weight_decay = 0.0 if SANITY_CHECK else 0.1 
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95))

        steps_per_epoch = math.ceil(self.dataset_size / self.batch_size)
        
        # PyTorch Lightning handles overfitting natively via the Trainer.
        total_steps = self.epochs * steps_per_epoch
        warmup_steps = int(0.03 * total_steps) # Change if you want the warmup to be longer/shorter.

        warmup_lrs = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=max(1, warmup_steps))
        cosine_decay_lrs = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=1e-6)
        
        # Failsafe to avoid scheduler crash if warmup_steps == 0.
        milestone = max(1, round(warmup_steps/total_steps) + 1) if total_steps > 0 else 1
        scheduler = SequentialLR(optimizer, schedulers=[warmup_lrs, cosine_decay_lrs], milestones=[milestone])

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}

class MusicDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, collator, batch_size, num_workers):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.collator = collator
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.collator, 
                          num_workers=self.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collator, 
                          num_workers=self.num_workers, pin_memory=True, shuffle=False)

def main():
    print("--- Training Pipeline ---")
    
    tokenizer = TSD(params=DATABASE / "tokenizer.json")
    # Use DataCollator for Dynamic Padding
    # IMPORTANT : Qwen3 already shifts labels internally, so setting it here to False is MANDATORY.
    # Set it to True ONLY if your model DOES NOT shift labels internally.
    collator = DataCollator(pad_token_id=tokenizer["PAD_None"], copy_inputs_as_labels=True, shift_labels=False, labels_pad_idx=-100)

    train_files = list((DATABASE / "midi_train").rglob("*.mid")) + list((DATABASE / "midi_train").rglob("*.midi"))
    val_files = list((DATABASE / "midi_val").rglob("*.mid")) + list((DATABASE / "midi_val").rglob("*.midi"))
    
    CACHE_DIR = DATABASE / "cache_tensors"
    CACHE_DIR.mkdir(exist_ok=True)

    # Cache naming management
    suffix = "_sanity" if SANITY_CHECK else "_full"
    train_cache = CACHE_DIR / f"train_cache{suffix}.pt"
    val_cache = CACHE_DIR / f"val_cache{suffix}.pt"

    ckpt_path, resume = None, False
    checkpoint_dir = OUTPUT_DIR / "checkpoints"
    if not SANITY_CHECK and checkpoint_dir.exists() and list(checkpoint_dir.glob("*.ckpt")):
        ckpt_path = str(max(checkpoint_dir.glob("*.ckpt"), key=lambda x: x.stat().st_mtime))
        resume = True
        print(f"Resuming from checkpoint: {ckpt_path}")

    # Loading via ram_dataset (see respective file for more info).
    # If unable to use, use DatasetMIDI instead and uncomment the import.
    train_dataset = RAMMidiDataset(
        files_paths=[train_files[0]] if SANITY_CHECK else train_files,
        tokenizer=tokenizer, max_seq_len=SEQ_LEN, cache_path=train_cache
    )
    val_dataset = RAMMidiDataset(
        files_paths=[val_files[0]] if SANITY_CHECK else val_files,
        tokenizer=tokenizer, max_seq_len=SEQ_LEN, cache_path=val_cache
    )

    DATASET_SIZE = 1 if SANITY_CHECK else len(train_dataset)
    EFFECTIVE_BATCH_SIZE = 1 if SANITY_CHECK else BATCH_SIZE * ACC_GRAD
    NUM_EPOCHS = 200 if SANITY_CHECK else round(STEPS * EFFECTIVE_BATCH_SIZE / max(1, DATASET_SIZE)) + 1

    datamodule = MusicDataModule(train_dataset, val_dataset, collator, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    
    model = Qwen3MusicModel(
        tokenizer=tokenizer, tokenizer_vocab_size=tokenizer.vocab_size, 
        learning_rate=LR, epochs=NUM_EPOCHS, dataset_size=DATASET_SIZE, 
        batch_size=EFFECTIVE_BATCH_SIZE, resume=resume
    )

    # Callbacks & Logger
    logger = TensorBoardLogger(str(ROOT_LOG_DIR), name="Sanity_Check" if SANITY_CHECK else "Qwen3")
    
    if SANITY_CHECK:
        callbacks_list = [TQDMProgressBar(), LearningRateMonitor('step'), InitialConditionAnalysis()]
    else:
        callbacks_list = [
            ModelCheckpoint(dirpath=str(checkpoint_dir), filename="Qwen-{epoch:02d}-{val_loss:.2f}", save_top_k=3, monitor="val_loss", mode="min", save_last=True),
            LearningRateMonitor(logging_interval='step'),
            TQDMProgressBar(refresh_rate=25),
            EarlyStopping(monitor="val_loss", patience=1, verbose=True),
            InitialConditionAnalysis(output_file=OUTPUT_DIR / "conditioning_report.txt") # See respective file for more info.
        ]

    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accumulate_grad_batches=ACC_GRAD,
        accelerator="auto",
        devices="auto",
        precision='bf16-mixed',  # Change depending on your hardware.
        gradient_clip_val=1.0,
        logger=logger,
        callbacks=callbacks_list,
        log_every_n_steps=1 if SANITY_CHECK else 50,
        check_val_every_n_epoch=None if SANITY_CHECK else 1, # Note: To validate mid-epoch, add the 'val_check_interval=0.5' argument instead.
        overfit_batches=1 if SANITY_CHECK else 0.0
    )

    print("\n Starting training...")
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path if not SANITY_CHECK else None)

if __name__ == "__main__":
    freeze_support()
    pl.seed_everything(SEED, workers=True)
    main()
