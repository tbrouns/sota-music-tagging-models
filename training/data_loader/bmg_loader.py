import os

import librosa
import numpy as np
import requests
import soundfile as sf
import torchaudio
import torchaudio.transforms as T

from torch.utils import data

from prosaic_common.config import get_cache_dir
from prosaic_common.queries import BigQuery
from prosaic_common.storage import GCP_BUCKETS
from prosaic_common.utils.utils_data import load_pickle
from prosaic_common.utils.utils_audio import load_audio_from_bucket

"""
Pre-requisite. First run:

    preprocessing/bmg_read.py
"""

class AudioFolder(data.Dataset):
    def __init__(self, batch_size=64, split="TRAIN", input_length=None, fs=16000):
        self.bucket = GCP_BUCKETS["songs"]
        self.client = BigQuery()
        self.cache_dir = os.path.join(get_cache_dir(), "bmg")
        self.mp3_dir = os.path.join(self.cache_dir, "mp3")
        os.makedirs(self.mp3_dir, exist_ok=True)
        self.fs = fs
        self.bmg_labels = load_pickle(
            os.path.join(self.cache_dir, "bmg_keywords.pkl")
        )
        self.num_keywords = self.bmg_labels.shape[0]
        self.batch_size = batch_size
        self.split = split
        self.input_length = input_length
        self.get_songlist()

    def __getitem__(self, index):
        # Download the song
        file_path = self.file_list[index]
        batch_index = index % self.batch_size
        load_path = os.path.join("harvest-extract", file_path)
        ext = os.path.splitext(file_path)[1]
        filename = f"{str(batch_index).zfill(3)}{ext}"
        save_path = os.path.join(self.mp3_dir, filename)
        x = load_audio_from_bucket(self.bucket, save_path, fs=self.fs)
        y_labels = np.zeros(self.num_keywords, dtype=float)
        if x is not None:
            audio_len = len(x)
            if audio_len < self.input_length:
                n_tiles = int(np.ceil(self.input_length / audio_len))
                x = np.tile(x, n_tiles)
                audio_len = len(x)
            # Get a random subsection of the song based on the model input length
            random_idx = int(
                np.floor(np.random.random(1) * (audio_len - self.input_length))
            )
            x_audio = np.array(x[random_idx : random_idx + self.input_length])
            tag_indices, tag_scores = self.file_dict[file_path]
            # Get the tag labels
            y_labels[tag_indices] = tag_scores
        else:
            # Return random signal
            x_audio = np.random.uniform(-1.0, 1.0, self.input_length)
        return x_audio.astype("float32"), y_labels.astype("float32")

    def __len__(self):
        return len(self.file_list)

    def get_songlist(self):
        pkl_path = os.path.join(self.cache_dir, f"bmg_{self.split.lower()}.pkl")
        if os.path.isfile(pkl_path):
            self.file_dict = load_pickle(pkl_path)
            self.file_list = list(self.file_dict.keys())
        else:
            raise f"Pickle path does not exist: {pkl_path}"

def get_audio_loader(
    root=None, batch_size=64, split="TRAIN", num_workers=0, input_length=None
):
    data_loader = data.DataLoader(
        dataset=AudioFolder(
            batch_size=batch_size, split=split, input_length=input_length
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )
    return data_loader
