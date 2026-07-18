"""
In-Memory MIDI Dataset Loader
This script implements a custom PyTorch Dataset designed to completely bypass I/O bottlenecks
during training. By leveraging parallel processing (joblib) and high-capacity RAM,
the entire dataset is tokenized, converted to PyTorch tensors, and cached directly in memory.
Note that the .pt file will roughly be 3 times heavier than your midi_val/midi_train folder,
so be sure you have enough RAM.
The main idea is to prioritize memory consumption over SSD read times to ensure the GPU
is constantly fed and never bottlenecked by data loading overhead.
"""

import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from joblib import Parallel, delayed


def process_one_file_direct(path, tokenizer, max_seq_len):
    """
    Loads a MIDI chunk and converts it to a tensor.
    Padding is not applied here (handled dynamically by the DataCollator).
    A safety clip is performed if the sequence exceeds max_seq_len.
    """
    try:
        # Tokenization via MidiTok
        tokens = tokenizer(path)

        if tokens is None or len(tokens) == 0:
            return []

        # ID extraction
        ids = tokens[0].ids if hasattr(tokens[0], "ids") else tokens[0]

        # FAILSAFE: Truncate if the bar-based split generated an oversized chunk, unlikely to happen though
        if len(ids) > max_seq_len:
            ids = ids[:max_seq_len]

        return [torch.LongTensor(ids)]

    except Exception as e:
        # In case of a corrupted file, return an empty list to prevent crashes, unlikely to happen since
        # the files went through extract_midi.py. Better safe than sorry though.
        print(f"Error on {path}: {e}")
        return []


class RAMMidiDataset(Dataset):
    def __init__(self, files_paths, tokenizer, max_seq_len, cache_path=None):
        """
        Optimized dataset for full in-memory loading.
        Prevents potential SSD bottlenecks during large-scale training.
        """
        self.examples = []

        # Cache management
        if cache_path and os.path.exists(cache_path):
            print(f"[CACHE] Immediate loading from {cache_path}...")
            self.examples = torch.load(cache_path)
            print(f"{len(self.examples)} sequences loaded into RAM.")
            return

        # Parallel loading
        print(f"Parallel tokenization of {len(files_paths)} files...")

        results_nested = Parallel(n_jobs=-1, backend="loky")(
            delayed(process_one_file_direct)(p, tokenizer, max_seq_len)
            for p in tqdm(files_paths, desc="Loading to RAM")
        )

        # Flattening the list of lists
        self.examples = [item for sublist in results_nested for item in sublist]

        print(f"Processing complete: {len(self.examples)} sequences ready.")

        # Saving cache
        if cache_path:
            print(f"Saving tensor cache: {cache_path}")
            torch.save(self.examples, cache_path)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Returning a dictionary for HuggingFace/MidiTok Collator compatibility
        return {"input_ids": self.examples[idx]}
