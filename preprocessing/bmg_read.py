import glob
import json
import logging
import os
import random

import librosa
import numpy as np
from tqdm import tqdm

from prosaic_common.config import get_cache_dir, get_config
from prosaic_common.queries import BigQuery
from prosaic_common.storage import GCP_BUCKETS
from prosaic_common.utils.logger import logger
from prosaic_common.utils.utils_data import (get_basename_no_extension,
                                             load_pickle, save_pickle)

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

"""
Create song list, add keywords, shuffle, do train/val/test split

Pre-requisite. First run:

    prosaic-research/music_tagging/match_tags.ipynb

Run from prosaic-research root:

     python -m third_party.sota_music_tagging_models.preprocessing.bmg_read

"""


class Processor:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = "music_tagging/config.ini"
        self.cfg = get_config(config_path=config_path)["match_tags"]
        self.bucket = GCP_BUCKETS["songs"]
        self.cache_dir = os.path.join(get_cache_dir(), "bmg")
        self.client = BigQuery()
        # TODO: add these PKL filenames to config.ini in prosaic_common config
        self.keyword_mapping = load_pickle(
            os.path.join(self.cache_dir, self.cfg["keyword_dict_mapped"])
        )
        self.keyword_cleaning = load_pickle(
            os.path.join(self.cache_dir, self.cfg["keyword_dict_cleaned"])
        )
        self.bmg_taxonomy = self.client.get_df_from_table_name("bmg_taxonomy")
        self.bmg_labels = load_pickle(
            os.path.join(self.cache_dir, "bmg_keywords.pkl")
        )
        self.missing_set_path = os.path.join(self.cache_dir, self.cfg["missing_files"])
        if os.path.isfile(self.missing_set_path):
            self.missing_set = load_pickle(self.missing_set_path)
            self.create_missing_set = False
        else:
            self.missing_set = set()
            self.create_missing_set = True
        self.data_dict = {}
        self.test_size = 2000
        self.val_size = 2000

    def save_split(self, filepaths, split="train"):
        split_dict = dict((k, self.data_dict[k]) for k in filepaths)
        save_path = os.path.join(self.cache_dir, f"bmg_{split}.pkl")
        save_pickle(save_path, split_dict)

    def get_songs_and_keywords(self):
        tracks = self.client.get_df_from_table_name("tracks")
        n_bmg_labels = self.bmg_labels.shape[0]
        keywords_dict = {}
        logger.info("Get the keywords for each song...")
        n_songs = len(tracks["file_path"])
        data_dict = {}
        for song_index in tqdm(range(n_songs)):
            multi_hot_vector = np.zeros(n_bmg_labels, dtype=float)
            filepath = tracks["file_path"][song_index]
            bucket_path = os.path.join("harvest-extract", filepath)
            tag_indices = []
            tag_scores = []
            if (not self.create_missing_set and filepath not in self.missing_set) or (
                self.create_missing_set
                and self.bucket.check_if_file_exists(bucket_path)
            ):
                # Get the raw free-text keywords
                record = tracks["record"][song_index]
                keywords_raw = json.loads(record)["track"]["keywords"]
                keyword_indices_for_song = []
                keywords_dict = {}
                for kw_raw in keywords_raw:
                    # First clean the raw keywords
                    if kw_raw in self.keyword_cleaning:
                        keywords_cleaned = self.keyword_cleaning[kw_raw]
                        # Then map the clean keywords to one of the labels in the BMG taxonomy
                        for kw_cleaned in keywords_cleaned:
                            if kw_cleaned in self.keyword_mapping:
                                keywords_mapped = self.keyword_mapping[kw_cleaned]
                                for kw, score in keywords_mapped.items():
                                    if (
                                        kw not in keywords_dict
                                        or score >= keywords_dict[kw]
                                    ):
                                        keywords_dict[kw] = score
                # Save the indices and corresponding scores
                for kw, score in keywords_dict.items():
                    index = np.nonzero(self.bmg_labels == kw)[0][0]
                    tag_indices.append(index)
                    tag_scores.append(score)
                if tag_indices:
                    self.data_dict[filepath] = (
                        np.array(tag_indices),
                        np.array(tag_scores),
                    )
            elif self.create_missing_set:
                self.missing_set.add(filepath)
        # Train/val/test split
        logger.info("Do train/val/test split...")
        filepath_list = list(self.data_dict.keys())
        random.shuffle(filepath_list)
        test_file_list = filepath_list[: self.test_size]
        val_file_list = filepath_list[-self.val_size :]
        train_file_list = filepath_list[self.test_size : -self.val_size]
        self.save_split(test_file_list, split="test")
        self.save_split(val_file_list, split="val")
        self.save_split(train_file_list, split="train")
        # Save set of missing files
        if self.create_missing_set:
            save_pickle(self.missing_set_path, self.missing_set)


if __name__ == "__main__":
    p = Processor()
    p.get_songs_and_keywords()
