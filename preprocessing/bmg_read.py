import glob
import os

import fire
import librosa
import numpy as np
import tqdm
import random

from prosaic_common.config import get_cache_dir
from prosaic_common.storage import GCP_BUCKETS
from prosaic_common.utils.utils_general import get_basename_no_extension

class Processor:
    def __init__(self):
        self.bucket = GCP_BUCKETS["songs"]
        self.cache_dir = os.path.join(get_cache_dir(), "bmg")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.fs = 16000

    def get_paths(self, data_path):
        self.files = self.bucket.get_list_of_files()
        self.files = [file in self.files if os.path.splitext(file)[1] == ".mp3"]
        self.npy_dir = os.path.join(data_path, "npy")
        os.makedirs(self.npy_dir, exist_ok=True)

    def get_npy(self, audio_path):
        """
            audio_path: relative path to audio file in bucket
        """
        save_path = os.path.join(self.cache_dir, os.path.basename(audio_path))
        #TODO: figure out whether this can be done without saving the file to disk
        self.bucket.download_to_file(load_path=audio_path, save_path=save_path)
        x, sr = librosa.core.load(save_path, sr=self.fs)
        os.remove(save_path)
        return x

    def iterate(self, data_path):
        self.get_paths(data_path)
        for audio_path in tqdm.tqdm(self.files):
            npy_dir =  os.path.join(self.npy_dir, os.path.dirname(audio_path))
            basename = get_basename_no_extension(audio_path)
            npy_path = os.path.join(npy_dir, f"{basename}.npy")
            if not os.path.exists(npy_path):
                try:
                    os.makedirs(npy_dir, exist_ok=True)
                    x = self.get_npy(audio_path)
                    np.save(open(npy_path, "wb"), x)
                except RuntimeError:
                    # some audio files are broken
                    print(audio_path)
                    continue
    
    def create_train_val_test_split(data_path, val_size=5000, test_size=5000):
        npy_file_list = glob.glob(os.path.join(self.npy_dir, "**", "*.npy"))
        random.shuffle(npy_file_list)
        test_file_list = npy_file_list[:test_size]
        val_file_list = npy_file_list[-val_size:]
        train_file_list = npy_file_list[test_size:-val_size]
        
                                
        
    

if __name__ == "__main__":
    p = Processor()
    fire.Fire({"run": p.iterate})
