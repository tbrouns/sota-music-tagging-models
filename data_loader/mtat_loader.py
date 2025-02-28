# coding: utf-8
import os

import numpy as np
from torch.utils import data


class AudioFolder(data.Dataset):
    def __init__(self, root, split, input_length=None):
        self.root = root
        self.split = split
        self.input_length = input_length
        self.get_songlist()
        self.binary = np.load("./../split/mtat/binary.npy")  # Multi-hot vector labels

    def __getitem__(self, index):
        npy, tag_binary = self.get_npy(index)
        return npy.astype("float32"), tag_binary.astype("float32")

    def get_songlist(self):
        """
        Returns list of strings. For example:

        index       string: `index`\t`song_path`
        ----------------------------------------------------------------------------------------------------------------
        [0000]      0   f/american_bach_soloists-(...)atas-01-bwv54__i_aria-30-59.mp3
        [....]      .
        [....]      .
        [0005]      11  e/steven_devine-portrait.(...)or_prelude_james_nares-0-29.mp3
        [....]      .
        [....]      .
        [0028]      115 d/electric_frankenstein-sick_songs-01-action_high-117-146.mp3
        [....]      .
        [....]      .
        [....]      etc.

        """

        if self.split == "TRAIN":
            self.fl = np.load("./../split/mtat/train.npy")
        elif self.split == "VALID":
            self.fl = np.load("./../split/mtat/valid.npy")
        elif self.split == "TEST":
            self.fl = np.load("./../split/mtat/test.npy")
        else:
            print("Split should be one of [TRAIN, VALID, TEST]")

    def get_npy(self, index):
        # Split at the tab delimiter, get the index and the song path
        ix, fn = self.fl[index].split("\t")
        # Loading the NPYs here, MP3s should be pre-processed first using `preprocessing/mtat_read.py`
        npy_path = os.path.join(self.root, "mtat", "npy", fn.split("/")[1][:-3]) + "npy"
        npy = np.load(npy_path, mmap_mode="r")
        # Get a random subsection of the song based on the model input length
        random_idx = int(np.floor(np.random.random(1) * (len(npy) - self.input_length)))
        npy = np.array(npy[random_idx : random_idx + self.input_length])
        # Get the corresponding multi-hot vector label
        tag_binary = self.binary[int(ix)]
        return npy, tag_binary

    def __len__(self):
        return len(self.fl)


def get_audio_loader(config, split="TRAIN"):
    data_loader = data.DataLoader(
        dataset=AudioFolder(
            root=config.data_path, split=split, input_length=config.input_length
        ),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=config.num_workers,
    )
    return data_loader
