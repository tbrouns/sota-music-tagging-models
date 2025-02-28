import argparse
import os

from prosaic_common.utils.logger import logger

from .solver import Solver


def train(config):
    # path for models
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)

    # import data loader
    if config.dataset == "mtat":
        from ..data_loader.mtat_loader import get_audio_loader
    elif config.dataset == "msd":
        from ..data_loader.msd_loader import get_audio_loader
    elif config.dataset == "jamendo":
        from ..data_loader.jamendo_loader import get_audio_loader
    elif config.dataset == "bmg":
        from ..data_loader.bmg_loader import get_audio_loader

    # audio length
    if config.model_type == "fcn" or config.model_type == "crnn":
        config.input_length = 29 * 16000
    elif config.model_type == "musicnn":
        config.input_length = 3 * 16000
    elif config.model_type in ["sample", "se", "short", "short_res"]:
        config.input_length = 59049
    elif config.model_type == "hcnn":
        config.input_length = 80000
    elif config.model_type == "attention":
        config.input_length = 15 * 16000

    # get data loder
    logger.info("Getting dataloader...")
    train_loader = get_audio_loader(
        config=config,
        split="TRAIN",
    )
    logger.info("Starting training...")
    solver = Solver(data_loader=train_loader, config=config)
    solver.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--dataset", type=str, default="mtat", choices=["mtat", "msd", "jamendo"]
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="fcn",
        choices=[
            "fcn",
            "musicnn",
            "crnn",
            "sample",
            "se",
            "short",
            "short_res",
            "attention",
            "hcnn",
        ],
    )
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use_tensorboard", type=int, default=1)
    parser.add_argument("--model_save_dir", type=str, default="./../models")
    parser.add_argument("--model_load_path", type=str, default=".")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--log_step", type=int, default=20)

    config = parser.parse_args()

    print(config)
    train(config)
