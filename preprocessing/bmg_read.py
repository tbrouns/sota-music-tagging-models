import glob
import os

import fire
import librosa
import numpy as np
from tqdm import tqdm
import random
import json

from prosaic_common.config import get_cache_dir
from prosaic_common.queries import BigQuery
from prosaic_common.storage import GCP_BUCKETS
from prosaic_common.utils.utils_data import get_basename_no_extension, save_pickle
from prosaic_common.utils.logger import logger


# Create song list, add keywords, shuffle, do train/val/test split


class Processor:
    def __init__(self):
        self.bucket = GCP_BUCKETS["songs"]
        self.cache_dir = os.path.join(get_cache_dir(), "bmg")
        self.client = BigQuery()
        self.data_dict = {}
        self.keyword_index = 0
        self.test_size = 5000
        self.val_size = 5000
        self.min_keywords = 1000

    def save_split(self, filepaths, split="train"):
        split_dict = dict((k, self.data_dict[k]) for k in filepaths)
        save_path = os.path.join(self.cache_dir, f"bmg_{split}.pkl")
        save_pickle(save_path, split_dict)

    def get_songs_and_keywords(self):
        tracks = self.client.get_df_from_table_name("tracks")
        keywords_dict = {}
        logger.info("Get the keywords for each song...")
        n_songs = len(tracks["file_path"])
        for song_index in tqdm(range(n_songs)):
            filepath = tracks["file_path"][song_index]
            record = tracks["record"][song_index]
            keywords = json.loads(record)["track"]["keywords"]
            keyword_indices_for_song = []
            for keyword in keywords:
                keyword = keyword.lower()
                if keyword not in keywords_dict:
                    keywords_dict[keyword] = self.keyword_index
                    self.keyword_index += 1
                keyword_indices_for_song.append(keywords_dict[keyword])
            if keyword_indices_for_song:
                self.data_dict[filepath] = keyword_indices_for_song
        # Count the occurrences of each keyword
        logger.info("Count the occurrences per keyword...")
        total_multi_hot_vector = np.zeros(self.keyword_index, dtype=int)
        for filepath, keyword_indices_for_song in self.data_dict.items():
            multi_hot_vector = np.zeros(self.keyword_index, dtype=int)
            multi_hot_vector[keyword_indices_for_song] = 1
            total_multi_hot_vector += multi_hot_vector
        # Save the multi hot vectors, filtering out the rare keywords
        logger.info("Filter out rare keywords...")
        keywords_to_keep = total_multi_hot_vector >= self.min_keywords
        files_to_delete = []
        for filepath, keyword_indices_for_song in self.data_dict.items():
            multi_hot_vector = np.zeros(self.keyword_index, dtype=int)
            multi_hot_vector[keyword_indices_for_song] = 1
            multi_hot_vector = multi_hot_vector[keywords_to_keep]
            if np.any(multi_hot_vector):
                self.data_dict[filepath] = np.nonzero(multi_hot_vector)[0]
            else:  # Mark the song for deletion if it doesn't have any common keywords
                files_to_delete.append(filepath)
        # Delete the songs marked for deletion
        for file in files_to_delete:
            if filepath in self.data_dict:
                del self.data_dict[filepath]
        # Train/val/test split
        logger.info("Do train/val/test split...")
        filepath_list = list(self.data_dict.keys())
        random.shuffle(filepath_list)
        test_file_list = filepath_list[:self.test_size]
        val_file_list = filepath_list[-self.val_size:]
        train_file_list = filepath_list[self.test_size:-self.val_size]
        self.save_split(test_file_list, split="test")
        self.save_split(val_file_list, split="val")
        self.save_split(train_file_list, split="train")
        # Save the keywords
        keywords = np.zeros(self.keyword_index, dtype=object)
        for keyword, index in keywords_dict.items():
            if keywords_to_keep[index]:
                keywords[index] = keyword
        keywords = keywords[keywords_to_keep]
        save_pickle(os.path.join(self.cache_dir, "bmg_keywords.pkl"), keywords)


if __name__ == "__main__":
    p = Processor()
    p.get_songs_and_keywords()