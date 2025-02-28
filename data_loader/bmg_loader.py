"""
Pre-requisite. First run:

    preprocessing/bmg_read.py
"""

import os

import numpy as np
from torch.utils import data

from prosaic_common.config import get_cache_dir
from prosaic_common.queries import BigQuery
from prosaic_common.storage import GCP_BUCKETS
from prosaic_common.utils.logger import logger
from prosaic_common.utils.utils_audio import load_audio_from_bucket
from prosaic_common.utils.utils_data import get_bmg_tags_for_category, load_pickle

from ..utils import get_pickle_filename


class AudioFolder(data.Dataset):
    def __init__(self, split="TRAIN", input_length=None, fs=16000, category=None):
        self.bucket = GCP_BUCKETS["songs"]
        self.bigquery = BigQuery()
        self.cache_dir = get_cache_dir()
        self.mp3_dir = os.path.join(self.cache_dir, "mp3")
        os.makedirs(self.mp3_dir, exist_ok=True)
        self.fs = fs
        logger.info(f"Picking tags for the {category} category...")
        self.category = category
        self.bmg_tags = get_bmg_tags_for_category(
            bigquery=self.bigquery, category=self.category
        )
        self.num_keywords = self.bmg_tags.shape[0]
        self.split = split
        self.input_length = input_length
        logger.info(f"There are {self.num_keywords} tags in the {self.split} data.")
        self.get_songlist()

    def __getitem__(self, index):
        # Download the song
        file_path = self.file_list[index]
        load_path = os.path.join("harvest-extract", file_path)
        x = load_audio_from_bucket(self.bucket, load_path, fs=self.fs)
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
            tag_indices = tag_indices.flatten()
            tag_scores = tag_scores.flatten()
            y_labels[tag_indices] = tag_scores
        else:
            # Return random signal
            x_audio = np.random.uniform(-1.0, 1.0, self.input_length)
        return x_audio.astype("float32"), y_labels.astype("float32")

    def __len__(self):
        return len(self.file_list)

    def get_songlist(self):
        filename = get_pickle_filename(split=self.split.lower(), category=self.category)
        pkl_path = os.path.join(self.cache_dir, filename)
        if os.path.isfile(pkl_path):
            self.file_dict = load_pickle(pkl_path)
            self.file_list = list(self.file_dict.keys())
        else:
            raise Exception(
                f"Pickle path does not exist: {pkl_path}. You need to run `preprocessing/bmg_read.py` first."
            )


def get_audio_loader(config, split="TRAIN"):
    data_loader = data.DataLoader(
        dataset=AudioFolder(
            split=split, input_length=config.input_length, category=config.category
        ),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=config.num_workers,
    )
    return data_loader
