# coding: utf-8
import csv
import datetime
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import tqdm
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from .model import (
    CNNSA,
    CRNN,
    FCN,
    HarmonicCNN,
    Musicnn,
    SampleCNN,
    SampleCNNSE,
    ShortChunkCNN,
    ShortChunkCNN_Res,
)

skip_files = set(
    [
        "TRAIISZ128F42684BB",
        "TRAONEQ128F42A8AB7",
        "TRADRNH128E0784511",
        "TRBGHEU128F92D778F",
        "TRCHYIF128F1464CE7",
        "TRCVDKQ128E0790C86",
        "TREWVFM128F146816E",
        "TREQRIV128F1468B08",
        "TREUVBN128F1468AC9",
        "TRDKNBI128F14682B0",
        "TRFWOAG128F14B12CB",
        "TRFIYAF128F14688A6",
        "TRGYAEZ128F14A473F",
        "TRIXPRK128F1468472",
        "TRAQKCW128F9352A52",
        "TRLAWQU128F1468AC8",
        "TRMSPLW128F14A544A",
        "TRLNGQT128F1468261",
        "TROTUWC128F1468AB4",
        "TRNDAXE128F934C50E",
        "TRNHIBI128EF35F57D",
        "TRMOREL128F1468AC4",
        "TRPNFAG128F146825F",
        "TRIXPOY128F14A46C7",
        "TROCQVE128F1468AC6",
        "TRPCXJI128F14688A8",
        "TRQKRKL128F1468AAE",
        "TRPKNDC128F145998B",
        "TRRUHEH128F1468AAD",
        "TRLUSKX128F14A4E50",
        "TRMIRQA128F92F11F1",
        "TRSRUXF128F1468784",
        "TRTNQKQ128F931C74D",
        "TRTTUYE128F4244068",
        "TRUQZKD128F1468243",
        "TRUINWL128F1468258",
        "TRVRHOY128F14680BC",
        "TRWVEYR128F1458A6F",
        "TRVLISA128F1468960",
        "TRYDUYU128F92F6BE0",
        "TRYOLFS128F9308346",
        "TRMVCVS128F1468256",
        "TRZSPHR128F1468AAC",
        "TRXBJBW128F92EBD96",
        "TRYPGJX128F1468479",
        "TRYNNNZ128F1468994",
        "TRVDOVF128F92DC7F3",
        "TRWUHZQ128F1451979",
        "TRXMAVV128F146825C",
        "TRYNMEX128F14A401D",
        "TREGWSL128F92C9D42",
        "TRJKZDA12903CFBA43",
        "TRBGJIZ128F92E42BC",
        "TRVWNOH128E0788B78",
        "TRCGBRK128F146A901",
    ]
)

TAGS = [
    "genre---downtempo",
    "genre---ambient",
    "genre---rock",
    "instrument---synthesizer",
    "genre---atmospheric",
    "genre---indie",
    "instrument---electricpiano",
    "genre---newage",
    "instrument---strings",
    "instrument---drums",
    "instrument---drummachine",
    "genre---techno",
    "instrument---guitar",
    "genre---alternative",
    "genre---easylistening",
    "genre---instrumentalpop",
    "genre---chillout",
    "genre---metal",
    "mood/theme---happy",
    "genre---lounge",
    "genre---reggae",
    "genre---popfolk",
    "genre---orchestral",
    "instrument---acousticguitar",
    "genre---poprock",
    "instrument---piano",
    "genre---trance",
    "genre---dance",
    "instrument---electricguitar",
    "genre---soundtrack",
    "genre---house",
    "genre---hiphop",
    "genre---classical",
    "mood/theme---energetic",
    "genre---electronic",
    "genre---world",
    "genre---experimental",
    "instrument---violin",
    "genre---folk",
    "mood/theme---emotional",
    "instrument---voice",
    "instrument---keyboard",
    "genre---pop",
    "instrument---bass",
    "instrument---computer",
    "mood/theme---film",
    "genre---triphop",
    "genre---jazz",
    "genre---funk",
    "mood/theme---relaxing",
]


def read_file(tsv_file):
    tracks = {}
    with open(tsv_file) as fp:
        reader = csv.reader(fp, delimiter="\t")
        next(reader, None)  # skip header
        for row in reader:
            track_id = row[0]
            tracks[track_id] = {
                "path": row[3].replace(".mp3", ".npy"),
                "tags": row[5:],
            }
    return tracks


class Solver(object):
    def __init__(self, data_loader, config, num_classes=50):
        # data loader
        self.data_loader = data_loader
        self.dataset = config.dataset
        self.data_path = config.data_path
        self.input_length = config.input_length
        self.num_classes = num_classes
        self.iteration_start = 0
        self.best_metric = 0.0
        self.threshold = config.threshold

        # training settings
        self.n_epochs = config.n_epochs
        self.lr = config.lr
        self.use_tensorboard = config.use_tensorboard
        self.reconst_loss = self.get_loss_function()

        # model path and step size
        self.model_save_dir = config.model_save_dir
        self.model_load_path = config.model_load_path
        self.log_step = config.log_step
        self.val_step = config.val_step
        self.batch_size = config.batch_size
        self.model_type = config.model_type

        # cuda
        self.is_cuda = torch.cuda.is_available()

        # Build model
        self.get_dataset()
        self.build_model()

        # Tensorboard
        self.writer = SummaryWriter(log_dir=config.log_dir)

    def get_dataset(self):
        if self.dataset == "mtat":
            self.valid_list = np.load("./../split/mtat/valid.npy")
            self.binary = np.load("./../split/mtat/binary.npy")
        elif self.dataset == "msd":
            train_file = os.path.join("./../split/msd", "filtered_list_train.cP")
            train_list = pickle.load(open(train_file, "rb"), encoding="bytes")
            val_set = train_list[201680:]
            self.valid_list = [
                value for value in val_set if value.decode() not in skip_files
            ]
            id2tag_file = os.path.join("./../split/msd", "msd_id_to_tag_vector.cP")
            self.id2tag = pickle.load(open(id2tag_file, "rb"), encoding="bytes")
        elif self.dataset == "jamendo":
            train_file = os.path.join(
                "./../split/mtg-jamendo", "autotagging_top50tags-validation.tsv"
            )
            self.file_dict = read_file(train_file)
            self.valid_list = list(read_file(train_file).keys())
            self.mlb = LabelBinarizer().fit(TAGS)
        elif self.dataset == "bmg":
            self.num_classes = self.data_loader.dataset.num_keywords
            from .data_loader.bmg_loader import get_audio_loader

            self.val_loader = get_audio_loader(
                batch_size=self.data_loader.batch_size,
                split="VAL",
                input_length=self.data_loader.dataset.input_length,
                num_workers=self.data_loader.num_workers,
            )

    def get_model(self):
        if self.model_type == "fcn":
            return FCN(n_class=self.num_classes)
        elif self.model_type == "musicnn":
            return Musicnn(dataset=self.dataset, n_class=self.num_classes)
        elif self.model_type == "crnn":
            return CRNN(n_class=self.num_classes)
        elif self.model_type == "sample":
            return SampleCNN(n_class=self.num_classes)
        elif self.model_type == "se":
            return SampleCNNSE(n_class=self.num_classes)
        elif self.model_type == "short":
            return ShortChunkCNN(n_class=self.num_classes)
        elif self.model_type == "short_res":
            return ShortChunkCNN_Res(n_class=self.num_classes)
        elif self.model_type == "attention":
            return CNNSA(n_class=self.num_classes)
        elif self.model_type == "hcnn":
            return HarmonicCNN(n_class=self.num_classes)

    def build_model(self):
        # model
        self.model = self.get_model()

        # cuda
        if self.is_cuda:
            self.model.cuda()

        # load pretrained model
        # TODO: load the model from the bucket
        if os.path.isfile(self.model_load_path):
            print(f"Loading model from: {self.model_load_path}...")
            self.load(self.model_load_path)
            basename = os.path.splitext(self.model_load_path)[0]
            iteration_start, best_metric = basename.split("_")[-2:]
            if iteration_start.isdigit() and best_metric.isdigit():
                self.iteration_start = int(iteration_start)
                self.best_metric = float(best_metric)
        else:
            print(f"Pre-trained model not found: {self.model_load_path}")

        # optimizers
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), self.lr, weight_decay=1e-4
        )

    def load(self, filename):
        S = torch.load(filename)
        if "spec.mel_scale.fb" in S.keys():
            self.model.spec.mel_scale.fb = S["spec.mel_scale.fb"]
        self.model.load_state_dict(S)

    def to_var(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def get_loss_function(self):
        return nn.BCELoss()

    def train(self):
        # Start training
        start_t = time.time()
        reconst_loss = self.get_loss_function()

        # drop_counter = 0
        n_samples = len(self.data_loader)

        # Iterate
        for epoch in range(self.n_epochs):
            # drop_counter += 1
            cumulative_loss = 0.0
            for ctr, (x, y) in enumerate(self.data_loader):

                ctr = ctr + 1
                iteration = self.iteration_start + epoch * n_samples + ctr

                # Forward
                x = self.to_var(x)
                y = self.to_var(y)
                out = self.model(x)

                # Backward
                loss = reconst_loss(out, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Log
                cumulative_loss += loss.item()
                if ctr % self.log_step == 0:
                    mean_loss = cumulative_loss / self.log_step
                    print_epoch = iteration // n_samples
                    print_ctr = (iteration % n_samples) + 1
                    self.print_log(print_epoch, print_ctr, mean_loss, start_t)
                    self.writer.add_scalar("Loss/train", mean_loss, iteration)
                    cumulative_loss = 0.0
                if ctr % self.val_step == 0:
                    # validation
                    print("Running validation ...")
                    self.validation(iteration)
                    print("Best metric:", self.best_metric)

            # # schedule optimizer
            # current_optimizer, drop_counter = self.opt_schedule(
            #     current_optimizer, drop_counter
            # )

        print(
            "[%s] Train finished. Elapsed: %s"
            % (
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                datetime.timedelta(seconds=time.time() - start_t),
            )
        )

    def opt_schedule(self, current_optimizer, drop_counter):
        # adam to sgd
        if current_optimizer == "adam" and drop_counter == 80:
            self.load(self.model_save_path)
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                0.001,
                momentum=0.9,
                weight_decay=0.0001,
                nesterov=True,
            )
            current_optimizer = "sgd_1"
            drop_counter = 0
            print("sgd 1e-3")
        # first drop
        if current_optimizer == "sgd_1" and drop_counter == 20:
            self.load(self.model_save_path)
            for pg in self.optimizer.param_groups:
                pg["lr"] = 0.0001
            current_optimizer = "sgd_2"
            drop_counter = 0
            print("sgd 1e-4")
        # second drop
        if current_optimizer == "sgd_2" and drop_counter == 20:
            self.load(self.model_save_path)
            for pg in self.optimizer.param_groups:
                pg["lr"] = 0.00001
            current_optimizer = "sgd_3"
            print("sgd 1e-5")
        return current_optimizer, drop_counter

    def save(self, filename):
        model = self.model.state_dict()
        torch.save({"model": model}, filename)

    def get_tensor(self, fn):
        # load audio
        if self.dataset == "mtat":
            npy_path = (
                os.path.join(self.data_path, "mtat", "npy", fn.split("/")[1][:-3])
                + "npy"
            )
        elif self.dataset == "msd":
            msid = fn.decode()
            filename = "{}/{}/{}/{}.npy".format(msid[2], msid[3], msid[4], msid)
            npy_path = os.path.join(self.data_path, filename)
        elif self.dataset == "jamendo":
            filename = self.file_dict[fn]["path"]
            npy_path = os.path.join(self.data_path, filename)

        raw = np.load(npy_path, mmap_mode="r")

        # split chunk
        length = len(raw)
        hop = (length - self.input_length) // self.batch_size
        x = torch.zeros(self.batch_size, self.input_length)
        for i in range(self.batch_size):
            x[i] = torch.Tensor(raw[i * hop : i * hop + self.input_length]).unsqueeze(0)
        return x

    def get_auc(self, est_array, gt_array):
        gt_array = (gt_array >= self.threshold).astype(int)
        # Check the number of unique values in the ground-truth
        # Should have at least one positive and one negative example for each tag
        keep = np.count_nonzero(np.diff(np.sort(gt_array, axis=0), axis=0), axis=0) >= 1
        gt_array = gt_array[:, keep]
        est_array = est_array[:, keep]
        # Calculate ROC and MAP
        roc_aucs = metrics.roc_auc_score(gt_array, est_array, average="macro")
        pr_aucs = metrics.average_precision_score(gt_array, est_array, average="macro")
        return roc_aucs, pr_aucs

    def print_log(self, epoch, ctr, loss, start_t):
        n_samples = len(self.data_loader)
        log_string = "[%s] Epoch [%d/%d] Iter [%d/%d] train loss: %.4f Elapsed: %s" % (
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            epoch + 1,
            self.n_epochs,
            ctr,
            n_samples,
            loss,
            datetime.timedelta(seconds=time.time() - start_t),
        )
        print(log_string)

    def validation(self, iteration):
        roc_auc, pr_auc, loss = self.get_validation_score(iteration)
        if roc_auc > self.best_metric:
            self.best_metric = roc_auc
            best_metric_str = "%.3f" % self.best_metric
            self.model_save_path = os.path.join(
                self.model_save_dir, f"best_model_{iteration}_{best_metric_str}.pth"
            )
            torch.save(
                self.model.state_dict(),
                self.model_save_path,
            )

    def get_score(self, x, y, ground_truth, losses, est_array, gt_array):
        out = self.model(x)
        loss = self.reconst_loss(out, y)
        losses.append(float(loss.data))
        out = out.detach().cpu().numpy().tolist()
        est_array.extend(out)
        gt_array.extend(ground_truth)
        return losses, est_array, gt_array

    def get_validation_score(self, iteration):
        est_array = []
        gt_array = []
        losses = []
        self.model.eval()
        if self.val_loader is not None:
            for x, y in self.val_loader:
                ground_truth = y.detach().cpu().numpy().tolist()
                x = self.to_var(x)
                y = self.to_var(y)
                losses, est_array, gt_array = self.get_score(
                    x, y, ground_truth, losses, est_array, gt_array
                )
        else:
            for line in tqdm.tqdm(self.valid_list):
                if self.dataset == "mtat":
                    ix, fn = line.split("\t")
                elif self.dataset == "msd":
                    fn = line
                    if fn.decode() in skip_files:
                        continue
                elif self.dataset == "jamendo":
                    fn = line

                # load and split
                x = self.get_tensor(fn)

                # ground truth
                if self.dataset == "mtat":
                    ground_truth = self.binary[int(ix)]
                elif self.dataset == "msd":
                    ground_truth = self.id2tag[fn].flatten()
                elif self.dataset == "jamendo":
                    ground_truth = np.sum(
                        self.mlb.transform(self.file_dict[fn]["tags"]), axis=0
                    )

                # forward
                x = self.to_var(x)
                y = torch.tensor(
                    [ground_truth.astype("float32") for i in range(self.batch_size)]
                ).cuda()

                losses, est_array, gt_array = self.get_score(
                    x, y, ground_truth, losses, est_array, gt_array
                )
        self.model.train()
        loss = np.mean(losses)
        est_array, gt_array = np.array(est_array), np.array(gt_array)
        roc_auc, pr_auc = self.get_auc(est_array, gt_array)
        print("loss: %.4f" % loss)
        print("roc_auc: %.4f" % roc_auc)
        print("pr_auc: %.4f" % pr_auc)
        self.writer.add_scalar("Loss/valid", loss, iteration)
        self.writer.add_scalar("AUC/ROC", roc_auc, iteration)
        self.writer.add_scalar("AUC/PR", pr_auc, iteration)
        return roc_auc, pr_auc, loss
