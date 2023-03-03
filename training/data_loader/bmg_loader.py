# Get `tracks` table from BigQuery
# Extract keywords from ["record"]["keywords"] field 
# Put this in dataloader class attributes, create song list
# Save keywords and song path
# Filter out songs that don't have any keywords etc.
# Shuffle the list ... 


# get item for data loader:
# Get batch of indices for song list
# Download the batch of song, load with librosa

import os
import numpy as np
from torch.utils import data
import librosa

from prosaic_common.config import get_cache_dir
from prosaic_common.utils.utils_data import load_pickle
from prosaic_common.storage import GCP_BUCKETS


class AudioFolder(data.Dataset):
    def __init__(self, batch_size=64, split="TRAIN", input_length=None, fs=16000):
        self.bucket = GCP_BUCKETS["songs"]
        self.cache_dir = os.path.join(get_cache_dir(), "bmg")
        self.mp3_dir = os.path.join(self.cache_dir, "mp3")
        os.makedirs(self.mp3_dir, exist_ok=True)
        self.fs = fs
        self.keywords = load_pickle(os.path.join(self.cache_dir, "bmg_keywords.pkl"))
        self.num_keyswords = len(self.keywords)
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
        #TODO: figure out whether this can be done without saving the file to disk
        self.bucket.download_to_file(load_path=load_path, save_path=save_path)
        x, sr = librosa.load(save_path, sr=self.fs)
        # Get a random subsection of the song based on the model input length
        random_idx = int(np.floor(np.random.random(1) * (len(x) - self.input_length)))
        x_audio = np.array(x[random_idx : random_idx + self.input_length])
        keyword_indices = self.file_dict[file_path]
        # Get the tag labels
        y_labels = np.zeros(self.num_keyswords, dtype=int)
        y_labels[keyword_indices] = 1
        return x_audio.astype("float32"), y_labels.astype("float32")
    
    def __len__(self):
        return len(self.file_list)
    
    def get_songlist(self):
        pkl_path = os.path.join(self.cache_dir, f"bmg_{self.split.lower()}.pkl")
        if os.path.isfile(pkl_path):
            self.file_dict = load_pickle(os.path.join(self.cache_dir, pkl_path))
            self.file_list = list(self.file_dict.keys())
        else:
            raise f"Pickle path does not exist: {pkl_path}"
            
    
    def get_paths(self):
        # Get all the paths to the audio files in the bucket
        self.files = self.bucket.get_list_of_files(delimiter="/", prefix="harvest-extract/")
        self.files = [f for f in self.files if os.path.splitext(f)[1] == ".mp3"]
        
    def get_npy(self, audio_path):
        """
            audio_path: relative path to audio file in bucket
        """
        save_path = os.path.join(self.cache_dir, os.path.basename(audio_path))
        #TODO: figure out whether this can be done without saving the file to disk
        self.bucket.download_to_file(load_path=audio_path, save_path=save_path)
        x, sr = librosa.load(save_path, sr=self.fs)
        os.remove(save_path)
        return x    
                       
            
def get_audio_loader(root, batch_size, split="TRAIN", num_workers=0, input_length=None):
    data_loader = data.DataLoader(
        dataset=AudioFolder(batch_size=batch_size, split=split, input_length=input_length),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )
    return data_loader
