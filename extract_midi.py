"""
MIDI Dataset Extraction & Tokenization Pipeline
This script handles the parsing, tokenization, and augmentation of raw MIDI files.
Architectural choice: Strict Time-Shift Duration (TSD) tokenization while discarding tokenizer training (BPE)
to preserve the fine-grained temporal resolution (Tempo Rubato) necessary for human-like musical reproduction.
"""

import shutil
from pathlib import Path
import random
from miditok import TSD, TokenizerConfig
from miditok.utils import split_files_for_training
from miditok.data_augmentation import augment_dataset
import pretty_midi
from tqdm import tqdm

# Global Parameters 
SEQ_LEN = 4096 # Preferably the same length as your model's context window
OVERLAP_BARS = 32 # Adjust depending on SEQ_LEN
MIDI_DIR = Path(__file__).resolve().parent / "raw_midis"
OUTPUT_DIR = Path(__file__).resolve().parent / "database"

# Tokenizer Configuration (Adjust depending on your goals) 
config = TokenizerConfig(
    pitch_range=(21, 109), 
    beat_res={(0, 4): 24, (4, 12): 8}, # High resolution on short notes
    num_velocities=127,
    special_tokens=["PAD", "BOS", "EOS", "MASK"],
    encode_ids_split="bar", 
    use_chords=False,
    use_rests=False,
    use_time_signatures=False,
    use_programs=False,
    program_changes=False,
    use_velocities=True,
    use_sustain_pedals=False,
    sustain_pedal_duration=True, # Sustain pedal modeled via note duration (relevant with TSD)
    use_tempos=True,
    num_tempos=64,
    tempo_range=(40, 250) 
)

tokenizer = TSD(config)  # Read Miditok's documentation for more info : https://miditok.readthedocs.io/en/latest/

def main():
    print("--- Initialization ---")
    
    if OUTPUT_DIR.exists(): 
        # Don't forget to save the tokenizer.json file if you plan to test different extraction methods
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    midi_files = list(MIDI_DIR.rglob("*.mid")) + list(MIDI_DIR.rglob("*.midi"))
    print(f"Raw MIDI files found: {len(midi_files)}")

    # ==============================================================================
    # [OPTIONAL MODULE] MIDI files validation
    # Since I used Google Magenta's Maestro dataset, which has already been verified,
    # checking the files would be pointless. However, if your dataset has been scraped from the internet, 
    # you might want to run the following script to avoid potentially empty or corrupted files.
    # ==============================================================================
    """
    print("Validating MIDI files integrity...")
    valid_midis = []
    for midi_path in tqdm(midi_files, desc="Verification"):
        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
            # Check that there are instruments and at least one note
            if len(midi.instruments) > 0 and any(len(inst.notes) > 0 for inst in midi.instruments):
                valid_midis.append(midi_path)
        except Exception:
            continue
    midi_files = valid_midis
    print(f"Valid files after cleaning: {len(midi_files)}")
    """

    # ==============================================================================
    # [OPTIONAL MODULE] Tokenizer training (e.g., BPE), cf. Miditok's documentation
    # Disabled because I prioritize strict temporal precision, thus I want the model
    # to only use elementary tokens. A large BPE vocabulary compresses the sequence 
    # but smooths out fine resolution (loss of rubato, cf. Byte Pair Encoding for Symbolic Music, Fradet et al., 2023),
    # which is irrelevant in ***my*** setting.
    # Enable if: Need to model global structures over very long durations 
    # (where compression takes precedence over temporal precision).
    # ==============================================================================
    """
    print("Training the Tokenizer...")
    tokenizer.train(
        vocab_size=20000,
        model="BPE",
        files_paths=midi_files
    )
    """

    tokenizer.save_pretrained(save_directory=OUTPUT_DIR)

    # Reproducible Train/Val split (90/10)
    random.seed(777)
    shuffled_files = midi_files.copy()
    random.shuffle(shuffled_files)

    # Test split omitted as qualitative evaluation is performed manually.
    # If you need one, this is the part you should modify
    num_train = round(0.9 * len(shuffled_files))
    train_files = shuffled_files[:num_train]
    val_files = shuffled_files[num_train:]

    # Chunk extraction
    for files_paths, subset_name in ((train_files, "train"), (val_files, "val")):
        subset_chunks_dir = OUTPUT_DIR / f"midi_{subset_name}"
        split_files_for_training(
            files_paths=files_paths,
            tokenizer=tokenizer,
            save_dir=subset_chunks_dir,
            max_seq_len=SEQ_LEN,
            num_overlap_bars=OVERLAP_BARS,
        )

        # ==============================================================================
        # [OPTIONAL MODULE] Data Augmentation
        # Depending on your dataset and model size, this script might be relevant.
        # Change this to artificially meet the "Chinchilla optimal scaling"
        # (dataset_size = 20 * model_size, cf. Training Compute-Optimal Large Language Models, Hoffmann et al., 2022)
        # if your dataset is not big enough.
        # This script might also solve an overfitting problem, though both augmentation and 
        # non-augmentation showed sufficient results for the same dataset and model size in my case.
        # ==============================================================================
        print(f"Augmenting {subset_name} set...")
        augment_dataset(
            subset_chunks_dir,
            pitch_offsets=[-3,-2,-1,1,2,3], 
            velocity_offsets=[-5,-2,2,5], 
            duration_offsets=[0.9,0.95,1.05,1.1], 
        )

    # Verification and Summary
    nb_train = len(list((OUTPUT_DIR / "midi_train").rglob("*.mid")) + list((OUTPUT_DIR / "midi_train").rglob("*.midi")))
    nb_val = len(list((OUTPUT_DIR / "midi_val").rglob("*.mid")) + list((OUTPUT_DIR / "midi_val").rglob("*.midi")))

    summary_text = (
        "--- EXTRACTION SUMMARY ---\n"
        f"Configured max sequence       : {SEQ_LEN} tokens\n"
        f"Overlap (bars)                : {OVERLAP_BARS}\n"
        f"Training chunks generated     : {nb_train}\n"
        f"Validation chunks generated   : {nb_val}\n"
        f"Final vocabulary size         : {tokenizer.vocab_size}\n"
        f"Tokenizer trained (BPE/etc.)  : {tokenizer.is_trained}\n"
    )

    # Console output
    print(f"\n{summary_text}")
    
    # Save to log file
    log_file_path = OUTPUT_DIR / "dataset_summary.txt"
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    
    print(f"Log saved to: {log_file_path}")

if __name__ == "__main__":
    main()