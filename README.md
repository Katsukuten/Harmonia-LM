# Harmonia-LM

Fully working pipeline to train a LLM for MIDI music generation. Includes tokenization and chunking of a MIDI dataset through Miditok, training and inference using Pytorch Lightning. Works with any CausalLM model found in Hugging Face's transformers python library. This project is my TIPE for french preparatory class.

## 1. Project Context: TIPE & Scientific Objectives

This project was developed within the framework of the French CPGE TIPE (Supervised Personal Research Project), under the annual theme: **"Cycles and Loops" (Cycles et Boucles)**.

Music is fundamentally cyclic (rhythmic loops, recurring motifs, harmonic progressions). While classic sequential models (RNNs/LSTMs) fail to maintain these loops over long periods, this project investigates whether a Large Language Model (LLM)—specifically the **Qwen3 architecture**—can assimilate musical syntax using NLP methodologies. 

By treating the [MAESTRO dataset](https://magenta.withgoogle.com/datasets/maestro) through a [Time-Shift Duration (TSD)](https://miditok.readthedocs.io/en/latest/tokenizations.html#:~:text=the%20whole%20music.-,TSD,-%C2%B6) tokenization, the objective is to validate the structural isomorphism between natural language and music, proving that the model can autonomously generate and maintain coherent musical cycles.

## 2. Audio Demonstrations

Those were made using the 4096-context-lenght model described below.

**1. Generation From Scratch (Unconditioned)**
A pure generation starting from a blank state, demonstrating the model's ability to create and sustain a rhythmic loop.

https://github.com/user-attachments/assets/6514ae80-f6dd-40e5-b3dd-61d3a67b22eb

**2. Primer Continuation (Beethoven's Moonlight Sonata 3rd Movement)**
The model is fed the beginning of the sonata and tasked to continue it, demonstrating its ability to adapt to a high-density semantic context.

https://github.com/user-attachments/assets/4dee7d7d-da97-4f46-a33b-8dd6e40754fa

## 3. Hardware Profiling & Model Configurations

To evaluate the impact of context length and data augmentation on the model's understanding of musical cycles, two distinct Qwen3 models were trained from scratch (with identical parameters: ~34M) on a local **RTX 5090 L (24GB VRAM)**.

### Model 1: The Baseline (2048 Context)
* **Context Window:** 2048 tokens
* **Dataset:** MAESTRO (No data augmentation)
* **Hardware Profiling:** Batch Size = 16, Gradient Accumulation = 8 (Effective Batch = 128).
* **VRAM Peak:** ~23 GB / 24 GB.

### Model 2: The Optimized Performer (4096 Context)
* **Context Window:** 4096 tokens
* **Dataset:** MAESTRO (Augmented with Pitch, Velocity, and Duration offsets)
* **Hardware Profiling:** Batch Size = 8, Gradient Accumulation = 16 (Effective Batch = 128).
* **VRAM Peak:** ~16 GB / 24 GB.

### Compute Optimizations (Bare-Metal)
* **Hardware-Aware Attention:** Native integration of Scaled Dot-Product Attention (SDPA / FlashAttention) to break the quadratic complexity of standard self-attention.
* **Mixed Precision:** Full BF16 implementation to prevent gradient underflow while maximizing Tensor Core throughput.
* **I/O Bottleneck Elimination:** A custom `joblib`-powered PyTorch Dataset pre-tokenizes and caches the entire dataset directly into RAM (96GB DDR5), bypassing SSD read latency during training.

## 4. Experimental Results & Interpretability

### 4.1 Training Dynamics (TensorBoard)
*(Add your TensorBoard Loss / Gradient Norm screenshots here)*
The training curves demonstrate the fast convergence of the Qwen3 architecture on symbolic music, validating our orthogonal initialization and GPT-2 variance scaling logic.
<p float="left">
  <img src="docs/tensorboard_loss.png" width="49%" />
</p>

### 4.2 Latent Space Topology (UMAP)
To prove the model mathematically maps musical syntax without prior bias, we project its embedding matrix into 2D and 3D spaces using UMAP.
**Post-Training State:** The model successfully warps its latent space to group tokens by functional family (Pitch, Velocity, Duration, TimeShift).
<p float="left">
  <img src="docs/umap_2d_trained.png" width="49%" />
  <img src="docs/umap_3d_trained.png" width="49%" /> 
</p>

### 4.3 Causal Attention Mapping
The causal attention heatmap from the final Transformer layer illustrates the "look-back" mechanism, showing exactly which prior tokens influenced the current generation step.
<p float="left">
  <img src="docs/attention_heatmap.png" width="60%" />
</p>

## 5. Repository Architecture

    Harmonia-LM/
    ├── .vscode/                     # VS Code workspace optimizations
    ├── docs/                        # Topological graphs, audio, and attention heatmaps
    ├── model/                       # Core engine
    │   ├── condition_callback.py    # Analyzes weight matrix condition numbers
    │   ├── inference_and_heatmap.py # Autoregressive sampling and visualization
    │   ├── ram_dataset.py           # In-memory parallelized dataset loader
    │   ├── training.py              # PyTorch Lightning training loop
    │   └── umap_projection.py       # Latent space extraction and UMAP mapping
    ├── raw_midis/                   # Target directory for raw .mid and .midi dataset
    ├── extract_midi.py              # TSD tokenization and data augmentation
    ├── README.md
    └── requirements.txt

## 6. Quickstart & Reproducibility

**1. Environment Setup**
    
    pip install -r requirements.txt

**2. Data Ingestion**
Place your `.mid` or `.midi` files inside the `raw_midis/` directory, then execute the extraction pipeline:
    
    python extract_midi.py

**3. Model Training**
The training script will automatically detect the tokenized database and cache it into RAM.
    
    python model/training.py

**4. Inference & Analysis**
    
    python model/inference_and_heatmap.py
    python model/umap_projection.py

## 7. References

This project was built upon the foundations laid by the following literature:

1. **Vaswani, A., et al. (2017).** *Attention Is All You Need.* [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
2. **Huang, C.-Z. A., et al. (2018).** *Music Transformer: Generating Music with Long-Term Structure.* [arXiv:1809.04281](https://arxiv.org/abs/1809.04281)
3. **Fradet, N., et al. (2021).** *MidiTok: A Python Package for MIDI Tokenization.* [arXiv:2310.17202](https://arxiv.org/abs/2310.17202)
4. **Team Qwen (2024).** *Qwen2.5 Technical Report.* [arXiv:2412.15115](https://arxiv.org/abs/2412.15115)
5. **Fradet, N., et al. (2023).** *Byte Pair Encoding for Symbolic Music.* [arXiv:2301.11975](https://arxiv.org/abs/2301.11975)
6. **Radford, A., et al. (2019).** *Language Models are Unsupervised Multitask Learners.* [OpenAI Blog](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
7. **McInnes, L., Healy, J., & Melville, J. (2018).** *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.* [arXiv:1802.03426](https://arxiv.org/abs/1802.03426)
