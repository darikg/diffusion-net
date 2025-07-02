from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Tuple, Any, Sequence

import numpy as np
import torch
from numpy import concatenate
from numpy.random import permutation
from pandas import Series
from tbparse import SummaryReader
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net  # noqa
from diffusion_net.utils import toNP   # noqa
from diffusion_net.layers import DiffusionNet   # noqa
from ga_dataset import GaDataset   # noqa


@dataclass
class Options:
    input_features: str
    data_file: Path
    data_dir: Path
    mesh_file_mode: str
    log_file: Path
    model_file: Path
    metadata_file: Path
    n_epoch: int
    channel: int | Sequence[int]
    k_eig: int = 128
    learning_rate: float = 1e-3
    decay_every = 50
    decay_rate = 0.5
    dropout: bool = False
    augment_random_rotate = False
    norm_verts: bool = False
    args: Any = None
    spike_window: tuple[float, float] = (0.07, 0.3)
    train_frac: float = 0.95
    weight_error: bool = True

    @staticmethod
    def for_timestamp(data_file: Path, stamp: str | None = None, **kwargs) -> Options:
        stamp = stamp or datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        data_dir = data_file.parent

        log_folder = data_dir / stamp
        log_folder.mkdir(parents=True, exist_ok=True)
        log_file = log_folder / f'diffnet_log_{stamp}.txt'
        model_file = log_folder / f'diffnet_model_{stamp}.pt'
        metadata_file = log_folder / f'metadata_{stamp}.pt'
        return Options(
            **kwargs,
            log_file=log_file,
            model_file=model_file,
            data_dir=data_dir,
            data_file=data_file,
            metadata_file=metadata_file,
        )

    @staticmethod
    def parse() -> Options:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--input_features",
            type=str,
            help="what features to use as input ('xyz' or 'hks') default: hks",
            default='hks',
        )
        parser.add_argument(
            '--file',
            type=str,
            help='the hdf file to load data from',
        )
        parser.add_argument(
            '--channel',
            type=int,
            help='which channel to use',
        )
        parser.add_argument(
            '--n-epoch',
            type=int,
            help='number of epochs',
        )
        parser.add_argument(
            '--dropout',
            action='store_true',
            help='enable dropout',
        )
        parser.add_argument(
            '--file-mode',
            type=str,
            help='filemode',
            default='simplified',
        )
        parser.add_argument(
            '--norm-verts',
            action='store_true',
            help='center and scale',
        )
        parser.add_argument(
            '--norm-response',
            action='store_true',
            help='normalize response to 0-1',
        )
        parser.add_argument(
            '--plot',
            action='store_true',
            help='plot last model',
        )
        parser.add_argument(
            '--window',
            nargs=2,
            type=int,
            default=(0.07, 0.3),
            help='spike window in seconds',
        )
        args = parser.parse_args()
        return Options.for_timestamp(
            data_file=Path(args.file),
            channel=args.channel,
            n_epoch=args.n_epoch,
            input_features=args.input_features,
            dropout=args.dropout,
            mesh_file_mode=args.file_mode,
            norm_verts=args.norm_verts,
            spike_window=args.window,
            args=args,
        )

    def init_log(self):
        logging.basicConfig(
            filename=self.log_file,
            level=logging.DEBUG,
            format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )

        # set up logging to console
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    def load_datasets(
            self,
            precalc_ops: bool = True,
            train_test_scenes: tuple[Sequence[int], Sequence] | None = None,
    ) -> Tuple[GaDataset, GaDataset]:
        scenes, responses, op_cache_dir = GaDataset.load_data(
            data_file=self.data_file,
            k_eig=self.k_eig,
            channel=self.channel,
            file_mode=self.mesh_file_mode,
            norm_verts=self.norm_verts,
            spike_window=self.spike_window,
            precalc_ops=precalc_ops,
        )

        if train_test_scenes:
            train_scenes, test_scenes = train_test_scenes
            train_idxs = np.isin(scenes.scene, train_scenes)
            test_idxs = np.isin(scenes.scene, test_scenes)
        else:
            n = len(scenes)
            idx = permutation(n)
            split = int(n * self.train_frac)
            train_idxs = idx < split
            test_idxs = idx >= split

        train_dataset, test_dataset = (
            GaDataset(
                df=scenes[idx],
                responses=responses[idx],
                root_dir=self.data_file.parent,
                k_eig=self.k_eig,
                op_cache_dir=op_cache_dir,
                normalize=self.norm_verts,
                file_mode=self.mesh_file_mode,
            )
            for idx in (train_idxs, test_idxs)
        )

        return train_dataset, test_dataset

    def make_model(self, n_channels_out: int = 1) -> DiffusionNet:
        return DiffusionNet(
            C_in=3 if self.input_features == 'xyz' else 16,
            C_out=n_channels_out,
            C_width=64,
            N_block=4,
            last_activation=None,
            outputs_at='global_mean',
            dropout=self.dropout,
        )

    def experiment(self, train_dataset: GaDataset, test_dataset: GaDataset) -> Experiment:
        device = torch.device('cuda:0')
        n_channels = 1 if isinstance(self.channel, int) else len(self.channel)
        model = self.make_model(n_channels_out=n_channels)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        folder = self.log_file.parent
        writer = SummaryWriter(log_dir=str(folder), flush_secs=10) if folder else None

        return Experiment(
            opts=self,
            model=model,
            device=device,
            optimizer=optimizer,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            writer=writer,
        )


@dataclass
class Experiment:
    opts: Options
    model: DiffusionNet
    device: torch.device
    optimizer: torch.optim.Adam
    train_dataset: GaDataset
    test_dataset: GaDataset
    writer: SummaryWriter

    def load_item(self, data):
        verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels = data
        verts = verts.to(self.device)
        faces = faces.to(self.device)
        _frames = frames.to(self.device)
        mass = mass.to(self.device)
        L = L.to(self.device)
        evals = evals.to(self.device)
        evecs = evecs.to(self.device)
        gradX = gradX.to(self.device)
        gradY = gradY.to(self.device)
        labels = labels.to(self.device)

        # if augment_random_rotate:
        #     verts = diffusion_net.utils.random_rotate_points(verts)

        # Construct features
        if self.opts.input_features == 'xyz':
            features = verts
        elif self.opts.input_features == 'hks':
            features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)
        else:
            raise NotImplementedError(self.opts.input_features)

        # Apply the model
        preds = self.model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
        return labels, preds

    def train_epoch(self, loader: DataLoader, epoch: int) -> float:
        # Returns mean training loss
        if epoch > 0 and epoch % self.opts.decay_every == 0:
            self.opts.learning_rate *= self.opts.decay_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.opts.learning_rate

        self.model.train()
        self.optimizer.zero_grad()
        losses = []

        for data in tqdm(loader):
            labels, preds = self.load_item(data)
            err = torch.square(preds - labels)
            if self.opts.weight_error:
                err = err * labels

            loss = torch.mean(err)
            losses.append(toNP(loss))
            loss.backward()

            # Step the optimizer
            self.optimizer.step()
            self.optimizer.zero_grad()

        return np.mean(losses)

    def test(self, loader):
        # Returns mean loss
        self.model.eval()
        losses = []

        with torch.no_grad():
            for data in tqdm(loader):
                labels, preds = self.load_item(data)
                loss = torch.mean(torch.square(preds - labels))
                losses.append(toNP(loss))

        return np.mean(losses)

    def predict(self, loader):
        self.model.eval()
        obs, preds = [], []

        with torch.no_grad():
            for data in tqdm(loader):
                _obs, _preds = self.load_item(data)
                obs.append(_obs.cpu().numpy())
                preds.append(_preds.cpu().numpy())

        return concatenate(obs), concatenate(preds)


def main():
    logger = logging.getLogger(__name__)
    opts = Options.for_timestamp(
        # data_file=Path(r"D:\resynth\run_51_52\1k_faces\run00051_resynth.hdf"),
        # channel=(0, 2, 29, 5, 17, 23, 14, 31, 18, 30, 7, 25, 3, 9),
        # data_file=Path(r"D:\resynth\run_48_49\1k_faces\run00048_resynth.hdf"),
        # channel=(14, 17, 29, 23, 2, 0, 13, 31, 3, 26, 28, 9, 20, 11, 18),
        # data_file=Path(r"D:\resynth\run_09_10\1k_faces\run00009_resynth.hdf"),
        # channel=(29, 2, 19, 31, 0, 23, 12, 14, 18, 8),
        data_file=Path(r"D:\resynth\run_42_43\1k_faces\run00042_resynth.hdf"),
        channel=(14, 17, 29, 23, 2, 0, 13, 31, 3, 26, 28, 9, 20, 11, 18),
        n_epoch=250,
        input_features='hks',
        dropout=True,
        mesh_file_mode='simplified',
        norm_verts=False,
        spike_window=(0.07, 0.75),
        train_frac=0.95,
        weight_error=False,
    )
    opts.init_log()
    train_dataset, test_dataset = opts.load_datasets(precalc_ops=False)
    exp = opts.experiment(train_dataset=train_dataset, test_dataset=test_dataset)

    metadata = dict(
        opts=opts,
        train_scenes=exp.train_dataset.df.scene.values,
        test_scenes=exp.test_dataset.df.scene.values,
    )
    torch.save(metadata, opts.metadata_file)

    train_loader = DataLoader(exp.train_dataset, batch_size=None, shuffle=True)
    test_loader = DataLoader(exp.test_dataset, batch_size=None)

    for epoch in range(opts.n_epoch):
        train_loss = exp.train_epoch(train_loader, epoch)
        exp.writer.add_scalar(f'loss/train', train_loss, epoch)
        test_loss = exp.test(test_loader)
        exp.writer.add_scalar(f'loss/test', test_loss, epoch)
        logger.debug(f"Epoch {epoch}: Train: {train_loss:.5e}  Test: {test_loss:.5e}")

    logger.debug("Saving last model to %s", opts.model_file)
    torch.save(exp.model.state_dict(), opts.model_file)


if __name__ == '__main__':
    main()


class Reader:
    def __init__(self, folder: Path):
        self.folder = folder
        self.reader = SummaryReader(str(folder))

    @property
    def log_path(self) -> Path:
        return Path(self.reader.log_path)

    @cached_property
    def hparams(self) -> Series:
        hparams = self.reader.hparams.set_index('tag').unstack().reset_index(level=0, drop=True)
        hparams['logfile'] = self.log_path.name
        return hparams

    @cached_property
    def scalars(self):
        return self.reader.scalars

    @lru_cache
    def scalar(self, tag):
        df = self.scalars
        df = df[df.tag == tag]
        return df.step, df.value

    def format_hparams(self, tags=None):
        df = self.reader.hparams.set_index('tag')
        if not tags:
            tags = [k for k, v in df.items() if len(set(v)) > 1 and k != 'logfile']

        return ', '.join(f'{k} = {df.loc[k].value}' for k in tags)
