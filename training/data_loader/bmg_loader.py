import os

import librosa
import numpy as np
import requests
import soundfile as sf
import torchaudio
import torchaudio.transforms as T
from miniaudio import SampleFormat, decode
from torch.utils import data

from prosaic_common.config import get_cache_dir
from prosaic_common.queries import BigQuery
from prosaic_common.storage import GCP_BUCKETS
from prosaic_common.utils.utils_data import load_pickle


class AudioFolder(data.Dataset):
    def __init__(self, batch_size=64, split="TRAIN", input_length=None, fs=16000):
        self.bucket = GCP_BUCKETS["songs"]
        self.client = BigQuery()
        self.cache_dir = os.path.join(get_cache_dir(), "bmg")
        self.mp3_dir = os.path.join(self.cache_dir, "mp3")
        os.makedirs(self.mp3_dir, exist_ok=True)
        self.fs = fs
        self.bmg_taxonomy = self.client.get_df_from_table_name("bmg_taxonomy")
        self.num_keywords = self.bmg_taxonomy.shape[0]
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
        audio_bytes = self.bucket.download_to_bytes(load_path=load_path)
        try:
            x = decode(
                audio_bytes,
                nchannels=1,
                sample_rate=self.fs,
                output_format=SampleFormat.FLOAT32,
            )
            x = np.array(x.samples)
        except Exception as e:
            x = None
            print(e)
            print(f"Could not load: {load_path}!")
        y_labels = np.zeros(self.num_keywords, dtype=int)
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

    def get_paths(self):
        # Get all the paths to the audio files in the bucket
        self.files = self.bucket.get_list_of_files(
            delimiter="/", prefix="harvest-extract/"
        )
        self.files = [f for f in self.files if os.path.splitext(f)[1] == ".mp3"]

    def get_npy(self, audio_path):
        """
        audio_path: relative path to audio file in bucket
        """
        save_path = os.path.join(self.cache_dir, os.path.basename(audio_path))
        # TODO: figure out whether this can be done without saving the file to disk
        self.bucket.download_to_file(load_path=audio_path, save_path=save_path)
        x, sr = librosa.load(save_path, sr=self.fs)
        os.remove(save_path)
        return x


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
