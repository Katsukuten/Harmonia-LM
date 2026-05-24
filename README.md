# Harmonia-LM

Fully working pipeline to train a LLM for MIDI music generation. Includes tokenization and chunking of a MIDI dataset through Miditok, training and inference using Pytorch Lightning. Works with any CausalLM model found in Hugging Face's transformers python library. This project is my TIPE for french preparatory class.

## Project Context

This project was developed within the framework of the French CPGE TIPE (Supervised Personal Research Project), under the annual theme: **"Cycles and Loops" (Cycles et Boucles)**.

Music is fundamentally cyclic (rhythmic loops, recurring motifs, harmonic progressions). While classic sequential models (RNNs/LSTMs) fail to maintain these loops over long periods, this project investigates whether a Large Language Model (LLM)—specifically the **Qwen3 architecture**—can assimilate musical syntax using NLP methodologies. 

By treating the [MAESTRO dataset](https://magenta.withgoogle.com/datasets/maestro) through a [Time-Shift Duration (TSD)](https://miditok.readthedocs.io/en/latest/tokenizations.html#:~:text=the%20whole%20music.-,TSD,-%C2%B6) tokenization, the objective is to validate the structural isomorphism between natural language and music, proving that the model can autonomously generate and maintain coherent musical cycles.

*Note: I plan to explain my reasoning and the technical pipeline further in an upcoming YouTube video.*

## Audio Demonstrations

Those were made using the 4096-context-length model described further below.

**1. Primer Continuation (Debussy's Clair de Lune)**
The model is fed the beginning of the piece and tasked to continue it. This piece sits perfectly within the high-density distribution center of the dataset. The model successfully interpolates the harmonic progressions and sustains the flow to some extent.
<br>
*(Insert Debussy Video Here)*
<br>

**2. Edge-of-Distribution Case (Beethoven's Moonlight Sonata 3rd Movement)**
The model is fed a highly complex primer. This movement features extreme rhythmic velocity and note density, pushing the model to the absolute boundaries of the MAESTRO dataset distribution. It highlights the model's struggle with tempos and densities rarely seen during training.
<br>
https://github.com/user-attachments/assets/4dee7d7d-da97-4f46-a33b-8dd6e40754fa
<br>

**3. Generation From Scratch**
A pure generation starting from a blank state (`BOS`). While not recommended—due to the immense variety of possible starting sequences in the dataset leading to initial noise (noticceable at the beginning of the piece), the model eventually converges toward a specific style. This unconditioned output represents a "statistical mean" of the dataset: a relatively slow tempo with conservative harmonic choices, corroborating the boundaries observed in the Beethoven demonstration.
<br>
https://github.com/user-attachments/assets/6514ae80-f6dd-40e5-b3dd-61d3a67b22eb
<br>

## Inference & Model Engineering  

Autoregressive generation in symbolic music reveals a strict dependency on context density. The decoding parameters must be tuned specifically to avoid deterministic loops or atonal structural collapse. The following are the parameters used for the aforementionned pieces :

* **Repetition Penalty disabled (1.0):** Unlike natural language, music is inherently fractal and cyclic. Any penalty forces the model to flee into chaotic dissonance to avoid repeating previous notes. Activating it won't break anything but proves irrelevant in this setting.
* **Temperature (0.85 - 0.90):** Lowered from standard NLP defaults to restrict the model's entropy, forcing it to adhere strictly to the harmonic structures learned in the dataset.
* **Top-K (20) & Top-P (0.95):** A tight truncation boundary to discard the long tail of unmapped, dissonant notes.

Model specifications :

* **RoPE (Rotary Positional Embeddings) Scaling:** Models equipped with RoPE Scaling (e.g, Qwen3) can make use of it to generate past their maximum context window. Though its utility decreases the larger the native context window is as it would outlength most of the pieces in any dataset.
* **MoE (Mixture of Experts):** Training a MoE model (e.g, Qwen3MoE) with this pipeline is entirely possible and would in theory produce better results as each expert will be specialized. However the model will have to have 1B+ parameters for it to be sufficiantly effective.

## Hardware Profiling & Model Configurations

To evaluate the impact of context length and data augmentation on the model's understanding of musical cycles, two distinct Qwen3 models were trained from scratch (~34M parameters) on a custom local setup.

**Hardware Setup:**
* **GPU:** NVIDIA RTX 5090 Laptop (23.5GB VRAM)
* **CPU:** AMD Ryzen 9955HX3D
* **RAM:** 96GB DDR5 5600MT/s

### Model 1: The Baseline (2048 Context)
* **Context Window:** 2048 tokens
* **Dataset:** MAESTRO (No data augmentation)
* **Hardware Profiling:** Batch Size = 16, Gradient Accumulation = 8 (Effective Batch = 128). VRAM Peak: ~23 GB.

### Model 2: The Optimized Performer (4096 Context)
* **Context Window:** 4096 tokens
* **Dataset:** MAESTRO (Augmented with Pitch, Velocity, and Duration offsets)
* **Hardware Profiling:** Batch Size = 8, Gradient Accumulation = 16 (Effective Batch = 128). VRAM Peak: ~16 GB.

### Compute Optimizations
* **Hardware-Aware Attention:** Native integration of Scaled Dot-Product Attention (SDPA) for mixed-precision BF16 operations, which're optimized on Nvidia's Blackwell architecture.
* **I/O Bottleneck Elimination:** The entire dataset is pre-tokenized and cached directly into the  RAM, completely bypassing SSD read latency during the training loop.

## Experimental Results 

### 4.1 Training Dynamics (TensorBoard)

The training curves demonstrate the fast convergence of the Qwen3 architecture on symbolic music, validating our orthogonal initialization and GPT-2 variance scaling logic.

### 4.2 Latent Space Topology (UMAP)
To prove the model mathematically maps musical syntax without prior bias, we project its embedding matrix into 2D and 3D spaces using UMAP.

**Prior to the training :** We can see the results of the orthogonal iniialization (same for both models since the tokenizer and seed are the same).
<img width="3600" height="2400" alt="umap_2d_initial" src="https://github.com/user-attachments/assets/9a13b22f-6e50-4946-917b-97ce42421005" />
<img width="3600" height="3000" alt="umap_3d_initial-True" src="https://github.com/user-attachments/assets/3237385e-9ae5-4f06-b8ea-2d5ba5b616f4" />

**Post-Training State:** The model successfully warps its latent space to group tokens by functional family (Pitch, Velocity, Duration, TimeShift).

For the **4096-context** model :
<img width="3600" height="3000" alt="umap_3d_initial-False" src="https://github.com/user-attachments/assets/4c4c0708-7c04-43f3-8fb8-3a12c2eec713" />
<img width="3600" height="2400" alt="umap_2d_initial-False" src="https://github.com/user-attachments/assets/2e5a3b2c-a245-4925-bea9-2e84e1085ed5" />


### 4.3 Causal Attention Mapping
The causal attention heatmap from the final Transformer layer illustrates the "look-back" mechanism, showing exactly which prior tokens influenced the current generation step.

## Repository Architecture

    Harmonia-LM/
    ├── .vscode/                     # VS Code workspace optimizations
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

## Quickstart & Reproducibility

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

## References

This project was built upon the foundations laid by the following literature:

1. **Vaswani, A., et al. (2017).** *Attention Is All You Need.* [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
2. **Huang, C.-Z. A., et al. (2018).** *Music Transformer: Generating Music with Long-Term Structure.* [arXiv:1809.04281](https://arxiv.org/abs/1809.04281)
3. **Fradet, N., et al. (2021).** *MidiTok: A Python Package for MIDI Tokenization.* [arXiv:2310.17202](https://arxiv.org/abs/2310.17202)
4. **Team Qwen (2024).** *Qwen2.5 Technical Report.* [arXiv:2412.15115](https://arxiv.org/abs/2412.15115)
5. **Fradet, N., et al. (2023).** *Byte Pair Encoding for Symbolic Music.* [arXiv:2301.11975](https://arxiv.org/abs/2301.11975)
6. **Radford, A., et al. (2019).** *Language Models are Unsupervised Multitask Learners.* [OpenAI Blog](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
7. **McInnes, L., Healy, J., & Melville, J. (2018).** *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.* [arXiv:1802.03426](https://arxiv.org/abs/1802.03426)
