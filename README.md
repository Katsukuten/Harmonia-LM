# Harmonia-LM

Fully working pipeline to train a LLM for MIDI music generation. Includes tokenization and chunking of a MIDI dataset through Miditok, training and inference using Pytorch Lightning. Works with any CausalLM model found in Hugging Face's transformers python library. This project is my TIPE for french preparatory class.

## Project Context

This project was developed within the framework of the French CPGE TIPE (Supervised Personal Research Project), under the annual theme: **"Cycles and Loops" (Cycles et Boucles)**.

Music is fundamentally cyclic (rhythmic loops, recurring motifs, harmonic progressions). While classic sequential models (RNNs/LSTMs) fail to maintain these loops over long periods, this project investigates whether a Large Language Model (LLM)—specifically the **Qwen3 architecture**—can assimilate musical syntax using NLP methodologies. 

By treating the [MAESTRO dataset](https://magenta.withgoogle.com/datasets/maestro) through a [Time-Shift Duration (TSD)](https://miditok.readthedocs.io/en/latest/tokenizations.html#:~:text=the%20whole%20music.-,TSD,-%C2%B6) tokenization, the objective is to validate the structural isomorphism between natural language and music, proving that the model can autonomously generate and maintain coherent musical cycles.

*Note: I plan to explain my reasoning and the technical pipeline further in an upcoming YouTube video.*

## Audio Demonstrations

Those were made using the 4096-context-length model described further below.

**- Primer Continuation (Debussy's Clair de Lune)**
The model is fed the beginning of the piece and tasked to continue it. This piece sits perfectly within the high-density distribution center of the dataset. The model successfully interpolates the harmonic progressions and sustains the flow to some extent. In a general setting, the longer the primer the better the continuation is.

https://github.com/user-attachments/assets/ec28495a-3c1c-4265-a2b6-3bbd945f43b4

**- Edge-of-Distribution Case (Beethoven's Moonlight Sonata 3rd Movement)**
The model is fed a highly complex primer. This movement features extreme rhythmic velocity and note density, pushing the model to the absolute boundaries of the MAESTRO dataset distribution. It highlights the model's struggle with tempos and densities rarely seen during training.

https://github.com/user-attachments/assets/22e38b23-b196-49d8-9517-df12073f9d8c


**- Generation From Scratch**
A pure generation starting from a blank state (`BOS`). While not recommended—due to the immense variety of possible starting sequences in the dataset leading to initial noise, the model eventually converges toward a specific style. This unconditioned output represents a "statistical mean" of the dataset: a relatively slow tempo with conservative harmonic choices, corroborating the boundaries observed in the Beethoven demonstration.

https://github.com/user-attachments/assets/2e97d8c9-86fb-4bd0-8b26-96b034014e98

## Inference & Model Engineering  

Autoregressive generation in symbolic music reveals a strict dependency on context density. The decoding parameters must be tuned specifically to avoid deterministic loops or atonal structural collapse. The following are the parameters used for the aforementioned pieces :

* **Repetition Penalty (1.05, 1.0 meaning no penalty):** Unlike natural language, music is inherently fractal and cyclic. Any penalty forces the model to flee into chaotic dissonance to avoid repeating previous notes. Giving a small repetition penalty blocks the model from entering in an obvious loop while not restricting its creativity.
* **Temperature (0.85 - 0.90):** Lowered from standard NLP defaults to restrict the model's entropy, forcing it to adhere strictly to the harmonic structures learned in the dataset.
* **Top-K (20) & Top-P (0.95):** A tight truncation boundary to discard the long tail of unmapped, dissonant notes.

Model specifications :

* **RoPE (Rotary Positional Embeddings) Scaling:** Models equipped with RoPE Scaling (e.g, Qwen3) can make use of it to generate past their maximum context window. Though its utility decreases the larger the native context window is as it would outlength most of the pieces in any dataset.
* **MoE (Mixture of Experts):** Training a MoE model (e.g, Qwen3MoE) with this pipeline is entirely possible and would in theory produce better results as each expert will be specialized. However the model will have to have 1B+ parameters for it to be sufficiently effective.

## Hardware Profiling & Model Configurations

To evaluate the impact of context length and data augmentation on the model's understanding of musical cycles, two distinct Qwen3 models were trained from scratch on a custom local setup.

**Hardware Setup:**
* **GPU:** NVIDIA RTX 5090 Laptop (24GB VRAM)
* **CPU:** AMD Ryzen 9955HX3D (16C/32T)
* **RAM:** 96GB DDR5 5600MT/s

### Model 1: The Baseline (2048 Context, ~29M Parameters)
* **Context Window:** 2048 tokens
* **Dataset:** MAESTRO (No data augmentation)
* **Hardware Profiling:** Batch Size = 16, Gradient Accumulation = 8 (Effective Batch = 128). VRAM Peak: ~23 GB.
* **Architecture:** 4 Attention Heads

### Model 2: The Optimized Performer (4096 Context, ~34M Parameters)
* **Context Window:** 4096 tokens
* **Dataset:** MAESTRO (Augmented with Pitch, Velocity, and Duration offsets)
* **Hardware Profiling:** Batch Size = 8, Gradient Accumulation = 16 (Effective Batch = 128). VRAM Peak: ~16 GB.
* **Architecture:** 8 Attention Heads 

### Compute Optimizations
* **Hardware-Aware Attention:** Native integration of Scaled Dot-Product Attention (SDPA) for mixed-precision BF16 operations, which are optimized on Nvidia's Blackwell architecture.
* **I/O Bottleneck Elimination:** The entire dataset is pre-tokenized and cached directly into the  RAM, completely bypassing SSD read latency during the training loop.

## Experimental Results 

### - Training Dynamics (TensorBoard)

EarlyStopping was used in the training with a patience of 1, but the saved model is the one with the lowest val_loss.

The 2048-context model took 2 epochs to converge before showing signs of overfitting.

<img width="1550" height="668" alt="image" src="https://github.com/user-attachments/assets/1a95a99b-f774-4565-a8d5-8811d27e29e1" />
<img width="1553" height="664" alt="image" src="https://github.com/user-attachments/assets/3da1659c-d471-433a-a9af-528bacc2d032" />
<img width="1560" height="663" alt="image" src="https://github.com/user-attachments/assets/3f99f415-3526-4e89-bf6a-e2ff9af8eaba" />

For the 4096-context model, the training curves demonstrate fast convergence (only one epoch), reaching a **Training Loss of 1.8** and a **Validation Loss of 3.3**. Thus validating the data augmentation.

<img width="1555" height="667" alt="image" src="https://github.com/user-attachments/assets/6a6a7a37-3420-4498-9dc8-c895f1d77826" />
<img width="1556" height="666" alt="image" src="https://github.com/user-attachments/assets/73908a68-5cf2-4300-a800-e9c0402f3bbd" />
<img width="1545" height="657" alt="image" src="https://github.com/user-attachments/assets/a220e27c-8242-4274-b610-b04f1c139df1" />

*Note on Validation Loss: In symbolic music modeling, validation loss naturally plateaus higher than in NLP. While a sentence has a strict grammatical continuation, a musical chord can resolve into dozens of aesthetically valid progressions. The Loss function penalizes the model for not predicting the *exact* original note of the composer, even if the model's choice is harmonically correct.*

### - Latent Space Topology (UMAP)
To prove the model mathematically maps musical syntax without prior bias, we project its embedding matrix into 2D and 3D spaces using UMAP.

**Prior to Training:** We can see the results of the orthogonal initialization (same for both models since the tokenizer and seed are the same).
<img width="4200" height="3000" alt="umap_2d_initial-True" src="https://github.com/user-attachments/assets/81d5a0b3-aabe-42f2-8b0d-a4f180cc0c3e" /><img width="4200" height="3000" alt="umap_3d_initial-True" src="https://github.com/user-attachments/assets/5119ca04-3f8a-4045-965e-09928e11fc31" />


**Post-Training State & Scale Discovery:** The models successfully warp their latent spaces to group tokens by functional family (Pitch, Velocity, Duration, TimeShift). More impressively, by applying Modulo 12 arithmetic to the Pitch tokens, we observe that the 4096-context model autonomously reconstructed the chromatic scale and isolated the **C Major scale** structurally, proving it "learned" music theory purely from statistical token co-occurrences. The dataset augmentation seems to be responsible for this, more than the widening of the context window.

For the **2048-context** model :

<img width="4200" height="3000" alt="umap_3d_initial-False-2048" src="https://github.com/user-attachments/assets/73a28007-fb29-4d58-a639-c962df444c49" />
<img width="4200" height="3000" alt="umap_2d_initial-False-2048" src="https://github.com/user-attachments/assets/bb0b7432-b138-42da-944a-30f4eb324575" />

For the **4096-context** model :

<img width="4200" height="3000" alt="umap_3d_initial-False" src="https://github.com/user-attachments/assets/a8c35a86-9409-4357-a0ac-893cdc9b8b89"/>
<img width="4200" height="3000" alt="umap_2d_initial-False" src="https://github.com/user-attachments/assets/16915dfb-0348-4a8f-b2d7-890bf2a5b96a"/>
<img width="2558" height="1492" alt="umap_scale" src="https://github.com/user-attachments/assets/1eecc969-d975-4db6-8fbb-cc27943b09fb"/>


### - Causal Attention Mapping
The causal attention heatmap from the final Transformer layer illustrates the "look-back" mechanism, showing exactly which prior tokens influenced the current generation step. The following is the heatmap of the last 100 tokens of the Debussy continuation further above, we can notice that the model seems to look back only to the few previous tokens, which might explain the loop at the end of the piece.

<img width="2380" height="2100" alt="primer-debussy-clair-d_len-4097_temp-0 95_topP-0 95_topK-0_rep-1 05_heatmap" src="https://github.com/user-attachments/assets/480cd7d1-b366-4149-ba87-2d91d87b3ff1" />

## Real-World Applications & DAW Integration

The empirical behavior of Harmonia-LM establishes that it should not be viewed as an autonomous end-to-end composer, but rather as an **intelligent co-pilot for human-in-the-loop composition**. 

Because the model excels at *continuation* (Primer) to some extent but struggles with *ex nihilo* generation, its optimal industrial application lies in its embedding as a plugin within professional Digital Audio Workstations (Ableton Live, FL Studio, Logic Pro). A human composer can write a 4-bar MIDI melody, and the local model can auto-regressively generate the accompanying bassline, harmony, or stylistic continuation based on the dataset's historical distribution.

Like most LLMs : the larger/more diverse the dataset is and larger the model is, the better the results will be.

> **Try it yourself:** An interactive deployment notebook is available to test the model's generation capabilities directly in the cloud.
> 👉 **[Run Harmonia-LM on Google Colab](#)**

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
