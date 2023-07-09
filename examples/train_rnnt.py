import random

import torch

from lrs3 import LRS3

import os

import torchaudio
import torchvision
from torch.utils.data import Dataset
import math

import torch
import random
from typing import List

import sentencepiece as spm
import torch
import torchvision
from data_module import LRS3DataModule
from lightning import Batch
from lightning_av import AVBatch


class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


class AdaptiveTimeMask(torch.nn.Module):
    def __init__(self, window, stride):
        super().__init__()
        self.window = window
        self.stride = stride

    def forward(self, x):
        cloned = x.clone()
        length = cloned.size(1)
        n_mask = int((length + self.stride - 0.1) // self.stride)
        ts = torch.randint(0, self.window, size=(n_mask, 2))
        for t, t_end in ts:
            if length - t <= 0:
                continue
            t_start = random.randrange(0, length - t)
            if t_start == t_start + t:
                continue
            t_end += t_start
            cloned[:, t_start:t_end] = 0
        return cloned


def _extract_labels(sp_model, samples: List):
    targets = [sp_model.encode(sample[-1].lower()) for sample in samples]
    lengths = torch.tensor([len(elem) for elem in targets]).to(dtype=torch.int32)
    targets = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(elem) for elem in targets],
        batch_first=True,
        padding_value=1.0,
    ).to(dtype=torch.int32)
    return targets, lengths


def _extract_features(video_pipeline, audio_pipeline, samples, args):
    raw_videos = []
    raw_audios = []
    for sample in samples:
        if args.md == "v":
            raw_videos.append(sample[0])
        if args.md == "a":
            raw_audios.append(sample[0])
        if args.md == "av":
            length = min(len(sample[0]) // 640, len(sample[1]))
            raw_audios.append(sample[0][: length * 640])
            raw_videos.append(sample[1][:length])

    if args.md == "v" or args.md == "av":
        videos = torch.nn.utils.rnn.pad_sequence(raw_videos, batch_first=True)
        videos = video_pipeline(videos)
        video_lengths = torch.tensor([elem.shape[0] for elem in videos], dtype=torch.int32)
    if args.md == "a" or args.md == "av":
        audios = torch.nn.utils.rnn.pad_sequence(raw_audios, batch_first=True)
        audios = audio_pipeline(audios)
        audio_lengths = torch.tensor([elem.shape[0] // 640 for elem in audios], dtype=torch.int32)
    if args.md == "v":
        return videos, video_lengths
    if args.md == "a":
        return audios, audio_lengths
    if args.md == "av":
        return audios, videos, audio_lengths, video_lengths


class TrainTransform:
    def __init__(self, sp_model_path: str, args):
        self.args = args
        self.sp_model = spm.SentencePieceProcessor(model_file=sp_model_path)
        self.train_video_pipeline = torch.nn.Sequential(
            FunctionalModule(lambda x: x / 255.0),
            torchvision.transforms.RandomCrop(88),
            torchvision.transforms.RandomHorizontalFlip(0.5),
            FunctionalModule(lambda x: x.transpose(0, 1)),
            torchvision.transforms.Grayscale(),
            FunctionalModule(lambda x: x.transpose(0, 1)),
            AdaptiveTimeMask(10, 25),
            torchvision.transforms.Normalize(0.421, 0.165),
        )
        self.train_audio_pipeline = torch.nn.Sequential(
            AdaptiveTimeMask(10, 25),
        )

    def __call__(self, samples: List):
        targets, target_lengths = _extract_labels(self.sp_model, samples)
        if self.args.md == "a":
            audios, audio_lengths = _extract_features(
                self.train_video_pipeline, self.train_audio_pipeline, samples, self.args
            )
            return Batch(audios, audio_lengths, targets, target_lengths)
        if self.args.md == "v":
            videos, video_lengths = _extract_features(
                self.train_video_pipeline, self.train_audio_pipeline, samples, self.args
            )
            return Batch(videos, video_lengths, targets, target_lengths)
        if self.args.md == "av":
            audios, videos, audio_lengths, video_lengths = _extract_features(
                self.train_video_pipeline, self.train_audio_pipeline, samples, self.args
            )
            return AVBatch(audios, videos, audio_lengths, video_lengths, targets, target_lengths)


class ValTransform:
    def __init__(self, sp_model_path: str, args):
        self.args = args
        self.sp_model = spm.SentencePieceProcessor(model_file=sp_model_path)
        self.valid_video_pipeline = torch.nn.Sequential(
            FunctionalModule(lambda x: x / 255.0),
            torchvision.transforms.CenterCrop(88),
            FunctionalModule(lambda x: x.transpose(0, 1)),
            torchvision.transforms.Grayscale(),
            FunctionalModule(lambda x: x.transpose(0, 1)),
            torchvision.transforms.Normalize(0.421, 0.165),
        )
        self.valid_audio_pipeline = torch.nn.Sequential(
            FunctionalModule(lambda x: x),
        )

    def __call__(self, samples: List):
        targets, target_lengths = _extract_labels(self.sp_model, samples)
        if self.args.md == "a":
            audios, audio_lengths = _extract_features(
                self.valid_video_pipeline, self.valid_audio_pipeline, samples, self.args
            )
            return Batch(audios, audio_lengths, targets, target_lengths)
        if self.args.md == "v":
            videos, video_lengths = _extract_features(
                self.valid_video_pipeline, self.valid_audio_pipeline, samples, self.args
            )
            return Batch(videos, video_lengths, targets, target_lengths)
        if self.args.md == "av":
            audios, videos, audio_lengths, video_lengths = _extract_features(
                self.valid_video_pipeline, self.valid_audio_pipeline, samples, self.args
            )
            return AVBatch(audios, videos, audio_lengths, video_lengths, targets, target_lengths)


class TestTransform:
    def __init__(self, sp_model_path: str, args):
        self.val_transforms = ValTransform(sp_model_path, args)

    def __call__(self, sample):
        return self.val_transforms([sample]), [sample]


def get_data_module(args, sp_model_path, max_frames=1800):
    train_transform = TrainTransform(sp_model_path=sp_model_path, args=args)
    val_transform = ValTransform(sp_model_path=sp_model_path, args=args)
    test_transform = TestTransform(sp_model_path=sp_model_path, args=args)
    return LRS3DataModule(
        args=args,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        max_frames=max_frames,
    )

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        steps_per_epoch: int,
        last_epoch=-1,
        verbose=False,
    ):
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            return [self._step_count / self.warmup_steps * base_lr for base_lr in self.base_lrs]
        else:
            decay_steps = self.total_steps - self.warmup_steps
            return [
                0.5 * base_lr * (1 + math.cos(math.pi * (self._step_count - self.warmup_steps) / decay_steps))
                for base_lr in self.base_lrs
            ]

def _load_list(args, *filenames):
    output = []
    length = []
    for filename in filenames:
        filepath = os.path.join(args.root_dir, "labels", filename)
        for line in open(filepath).read().splitlines():
            dataset, rel_path, input_length = line.split(",")[0], line.split(",")[1], line.split(",")[2]
            path = os.path.normpath(os.path.join(args.root_dir, dataset, rel_path[:-4] + ".mp4"))
            length.append(int(input_length))
            output.append(path)
    return output, length


def load_video(path):
    """
    rtype: torch, T x C x H x W
    """
    vid = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")[0]
    vid = vid.permute((0, 3, 1, 2))
    return vid


def load_audio(path):
    """
    rtype: torch, T x 1
    """
    waveform, sample_rate = torchaudio.load(path, normalize=True)
    return waveform.transpose(1, 0)


def load_transcript(path):
    transcript_path = path.replace("video_seg", "text_seg")[:-4] + ".txt"
    return open(transcript_path).read().splitlines()[0]


def load_item(path, md):
    if md == "v":
        return (load_video(path), load_transcript(path))
    if md == "a":
        return (load_audio(path), load_transcript(path))
    if md == "av":
        return (load_audio(path), load_video(path), load_transcript(path))


class LRS3(Dataset):
    def __init__(
        self,
        args,
        subset: str = "train",
    ) -> None:

        if subset is not None and subset not in ["train", "val", "test"]:
            raise ValueError("When `subset` is not None, it must be one of ['train', 'val', 'test'].")

        self.args = args

        if subset == "train":
            self._filelist, self._lengthlist = _load_list(self.args, "lrs3_train_transcript_lengths_seg16s.csv")
        if subset == "val":
            self._filelist, self._lengthlist = _load_list(self.args, "lrs3_test_transcript_lengths_seg16s.csv")
        if subset == "test":
            self._filelist, self._lengthlist = _load_list(self.args, "lrs3_test_transcript_lengths_seg16s.csv")

    def __getitem__(self, n):
        path = self._filelist[n]
        return load_item(path, self.args.md)

    def __len__(self) -> int:
        return len(self._filelist)

def _batch_by_token_count(idx_target_lengths, max_frames, batch_size=None):
    batches = []
    current_batch = []
    current_token_count = 0
    for idx, target_length in idx_target_lengths:
        if current_token_count + target_length > max_frames or (batch_size and len(current_batch) == batch_size):
            batches.append(current_batch)
            current_batch = [idx]
            current_token_count = target_length
        else:
            current_batch.append(idx)
            current_token_count += target_length

    if current_batch:
        batches.append(current_batch)

    return batches


class CustomBucketDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        lengths,
        max_frames,
        num_buckets,
        shuffle=False,
        batch_size=None,
    ):
        super().__init__()

        assert len(dataset) == len(lengths)

        self.dataset = dataset

        max_length = max(lengths)
        min_length = min(lengths)

        assert max_frames >= max_length

        buckets = torch.linspace(min_length, max_length, num_buckets)
        lengths = torch.tensor(lengths)
        bucket_assignments = torch.bucketize(lengths, buckets)

        idx_length_buckets = [(idx, length, bucket_assignments[idx]) for idx, length in enumerate(lengths)]
        if shuffle:
            idx_length_buckets = random.sample(idx_length_buckets, len(idx_length_buckets))
        else:
            idx_length_buckets = sorted(idx_length_buckets, key=lambda x: x[1], reverse=True)

        sorted_idx_length_buckets = sorted(idx_length_buckets, key=lambda x: x[2])
        self.batches = _batch_by_token_count(
            [(idx, length) for idx, length, _ in sorted_idx_length_buckets],
            max_frames,
            batch_size=batch_size,
        )

    def __getitem__(self, idx):
        return [self.dataset[subidx] for subidx in self.batches[idx]]

    def __len__(self):
        return len(self.batches)


class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform_fn):
        self.dataset = dataset
        self.transform_fn = transform_fn

    def __getitem__(self, idx):
        return self.transform_fn(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)


class LRS3DataModule(LightningDataModule):
    def __init__(
        self,
        *,
        args,
        train_transform,
        val_transform,
        test_transform,
        max_frames,
        batch_size=None,
        train_num_buckets=50,
        train_shuffle=True,
        num_workers=10,
    ):
        super().__init__()
        self.args = args
        self.train_dataset_lengths = None
        self.val_dataset_lengths = None
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.max_frames = max_frames
        self.batch_size = batch_size
        self.train_num_buckets = train_num_buckets
        self.train_shuffle = train_shuffle
        self.num_workers = num_workers

    def train_dataloader(self):
        datasets = [LRS3(self.args, subset="train")]

        if not self.train_dataset_lengths:
            self.train_dataset_lengths = [dataset._lengthlist for dataset in datasets]

        dataset = torch.utils.data.ConcatDataset(
            [
                CustomBucketDataset(
                    dataset,
                    lengths,
                    self.max_frames,
                    self.train_num_buckets,
                    batch_size=self.batch_size,
                )
                for dataset, lengths in zip(datasets, self.train_dataset_lengths)
            ]
        )

        dataset = TransformDataset(dataset, self.train_transform)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=None,
            shuffle=self.train_shuffle,
        )
        return dataloader

    def val_dataloadimport math

import torch


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        steps_per_epoch: int,
        last_epoch=-1,
        verbose=False,
    ):
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            return [self._step_count / self.warmup_steps * base_lr for base_lr in self.base_lrs]
        else:
            decay_steps = self.total_steps - self.warmup_steps
            return [
                0.5 * base_lr * (1 + math.cos(math.pi * (self._step_count - self.warmup_steps) / decay_steps))
                for base_lr in self.base_lrs
            ]er(self):
        datasets = [LRS3(self.args, subset="val")]

        if not self.val_dataset_lengths:
            self.val_dataset_lengths = [dataset._lengthlist for dataset in datasets]

        dataset = torch.utils.data.ConcatDataset(
            [
                CustomBucketDataset(
                    dataset,
                    lengths,
                    self.max_frames,
                    1,
                    batch_size=self.batch_size,
                )
                for dataset, lengths in zip(datasets, self.val_dataset_lengths)
            ]
        )

        dataset = TransformDataset(dataset, self.val_transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=self.num_workers)
        return dataloader

    def test_dataloader(self):
        dataset = LRS3(self.args, subset="test")
        dataset = TransformDataset(dataset, self.test_transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)
        return dataloader

import logging
import os
from argparse import ArgumentParser

import sentencepiece as spm
from average_checkpoints import ensemble
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from transforms import get_data_module


def get_trainer(args):
    seed_everything(1)

    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(args.exp_dir, args.experiment_name) if args.exp_dir else None,
        monitor="monitoring_step",
        mode="max",
        save_last=True,
        filename="{epoch}",
        save_top_k=10,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [
        checkpoint,
        lr_monitor,
    ]
    return Trainer(
        sync_batchnorm=True,
        default_root_dir=args.exp_dir,
        max_epochs=args.epochs,
        num_nodes=args.num_nodes,
        devices=args.gpus,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )


def get_lightning_module(args):
    sp_model = spm.SentencePieceProcessor(model_file=str(args.sp_model_path))
    if args.md == "av":
        from lightning_av import AVConformerRNNTModule

        model = AVConformerRNNTModule(args, sp_model)
    else:
        from lightning import ConformerRNNTModule

        model = ConformerRNNTModule(args, sp_model)
    return model


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--md",
        type=str,
        help="Modality",
        required=True,
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="Perform online or offline recognition.",
        required=True,
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        help="Root directory to LRS3 audio-visual datasets.",
        required=True,
    )
    parser.add_argument(
        "--sp-model-path",
        type=str,
        help="Path to SentencePiece model.",
        required=True,
    )
    parser.add_argument(
        "--pretrained-model-path",
        type=str,
        help="Path to Pretraned model.",
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        help="Directory to save checkpoints and logs to. (Default: './exp')",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Experiment name",
    )
    parser.add_argument(
        "--num-nodes",
        default=8,
        type=int,
        help="Number of nodes to use for training. (Default: 8)",
    )
    parser.add_argument(
        "--gpus",
        default=8,
        type=int,
        help="Number of GPUs per node to use for training. (Default: 8)",
    )
    parser.add_argument(
        "--epochs",
        default=55,
        type=int,
        help="Number of epochs to train for. (Default: 55)",
    )
    parser.add_argument(
        "--resume-from-checkpoint", default=None, type=str, help="Path to the checkpoint to resume from"
    )
    parser.add_argument("--debug", action="store_true", help="whether to use debug level for logging")
    return parser.parse_args()


def init_logger(debug):
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")


def cli_main():
    args = parse_args()
    init_logger(args.debug)
    model = get_lightning_module(args)
    data_module = get_data_module(args, str(args.sp_model_path))
    trainer = get_trainer(args)
    trainer.fit(model, data_module)

    ensemble(args)


if __name__ == "__main__":
    cli_main()