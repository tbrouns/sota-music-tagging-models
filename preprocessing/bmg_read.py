"""
Create song list, add keywords, shuffle, do train/val/test split

Pre-requisite. First run:

    prosaic-research/music_tagging/match_tags.ipynb

Run from prosaic-research root:

     python -m third_party.sota_music_tagging_models.preprocessing.bmg_read

"""


import json
import logging
import os
import random

import numpy as np
from tqdm import tqdm

from prosaic_common.config import get_cache_dir, get_config_and_combine
from prosaic_common.queries import BigQuery
from prosaic_common.storage import GCP_BUCKETS
from prosaic_common.utils.logger import logger
from prosaic_common.utils.utils_data import load_pickle, save_pickle

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


class Processor:
    def __init__(self, config_path=None):
        if config_path is None:
            # TODO: decouple this properly
            config_path = "data_processing/config.ini"
        self.cfg = get_config_and_combine(config_path=config_path)
        self.bucket = GCP_BUCKETS["songs"]
        self.bucket_artifacts = GCP_BUCKETS["artifacts"]
        self.cache_dir = get_cache_dir()
        self.bigquery = BigQuery()
        # Get the labels
        self.bmg_taxonomy = self.bigquery.get_df_from_table_name("bmg_taxonomy")
        self.bmg_labels = self.bmg_taxonomy["label"].to_numpy()
        # Get the PKL files from Google Storage
        self.keyword_mapping = self.download_pickle(
            self.cfg["match_tags"]["keyword_dict_mapped"]
        )
        self.keyword_cleaning = self.download_pickle(
            self.cfg["match_tags"]["keyword_dict_cleaned"]
        )
        if self.keyword_mapping is None or self.keyword_cleaning is None:
            raise Exception(
                "pkl files not found on storage. Run `data_processing/match_tags.ipynb` first"
            )
        self.missing_set = self.download_pickle(self.cfg["match_tags"]["missing_files"])
        if self.missing_set is not None:
            self.create_missing_set = False
            self.missing_set_path = None
        else:
            self.missing_set = set()
            self.create_missing_set = True
            self.missing_set_path = os.path.join(
                self.cache_dir, self.cfg["match_tags"]["missing_files"]
            )
        self.data_dict = {}
        self.test_size = 2000
        self.val_size = 2000

    def download_pickle(self, filename):
        save_path = self.bucket_artifacts.download_to_file(
            load_path=os.path.join("tagging", filename)
        )
        if save_path is not None:
            contents = load_pickle(save_path)
        else:
            contents = None
        return contents

    def get_pickle_filepath(self, split):
        return os.path.join(self.cache_dir, f"bmg_{split}.pkl")

    def save_split(self, filepaths, split="train"):
        split_dict = dict((k, self.data_dict[k]) for k in filepaths)
        save_pickle(self.get_pickle_filepath(split), split_dict)

    def load_train_val_test_split(self):
        splits = ["train", "val", "test"]
        save_path_dict = {}
        for split in splits:
            save_path = self.get_pickle_filepath(split=split)
            if os.path.isfile(save_path):
                save_path_dict[split] = save_path
            else:
                return None
        return save_path_dict

    def save_train_val_test_split(self):
        logger.info("Do train/val/test split...")
        filepath_list = list(self.data_dict.keys())
        random.shuffle(filepath_list)
        test_file_list = filepath_list[: self.test_size]
        val_file_list = filepath_list[-self.val_size :]
        train_file_list = filepath_list[self.test_size : -self.val_size]
        self.save_split(test_file_list, split="test")
        self.save_split(val_file_list, split="val")
        self.save_split(train_file_list, split="train")

    def get_tag_indices_and_scores(self):
        tracks_table_name = self.cfg["data"]["tracks_table"]
        logger.info(f"Loading {tracks_table_name} table...")
        tracks = self.bigquery.get_df_from_table_name(tracks_table_name)
        keywords_dict = {}
        logger.info("Get the keywords for each song...")

        n_songs = len(tracks["file_path"])
        for song_index in tqdm(range(n_songs)):
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

    def get_songs_and_keywords(self):

        save_path_dict = self.load_train_val_test_split()

        if save_path_dict is None:

            # Get the data dict
            self.get_tag_indices_and_scores()

            # Train/val/test split
            self.save_train_val_test_split()

            # Save set of missing files
            if self.create_missing_set:

                save_pickle(self.missing_set_path, self.missing_set)
                upload_path = os.path.join(
                    "tagging", os.path.basename(self.missing_set_path)
                )
                self.bucket_artifacts.upload_file(
                    self.missing_set_path, save_path=upload_path
                )


if __name__ == "__main__":
    p = Processor()
    p.get_songs_and_keywords()
