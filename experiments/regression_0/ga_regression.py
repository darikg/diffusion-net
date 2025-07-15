from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property, lru_cache
from itertools import product
from pathlib import Path
from typing import Tuple, Any, Sequence, Literal, Iterator

import numpy as np
import torch
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


def hparam_combinations(hparams: dict[str, Sequence[Any]]) -> Iterator[dict[str, Any]]:
    for vals in product(*hparams.values()):
        yield {k: v for k, v in zip(hparams.keys(), vals)}


@dataclass
class Metadata:
    opts: Options
    log_folder: Path
    model_file: Path
    metadata_file: Path

    input_features: str
    channel: int | Sequence[int]
    k_eig: int = 128
    learning_rate: float = 1e-3
    decay_every: int = 50
    decay_rate: float = 0.5
    dropout: bool = False
    n_faces: int | None = None
    spike_window: tuple[float, float] = (0.07, 0.3)
    weight_error: None | Literal['response', 'binned'] = 'binned'

    def load_datasets(
            self,
            precalc_ops: bool = True,
            train_test_scenes: tuple[Sequence[int], Sequence[int]] | None = None,
    ) -> Tuple[GaDataset, GaDataset]:
        scenes, responses, op_cache_dir, weights, fit_fns = GaDataset.load_data(
            data_file=self.opts.data_file,
            k_eig=self.k_eig,
            channel=self.channel,
            file_mode=self.opts.mesh_file_mode,
            norm_verts=False,
            spike_window=self.spike_window,
            precalc_ops=precalc_ops,
            weight_error=self.weight_error,
            n_faces=self.n_faces,
        )

        if train_test_scenes:
            train_scenes, test_scenes = train_test_scenes
            train_idxs = np.isin(scenes.scene, train_scenes)
            test_idxs = np.isin(scenes.scene, test_scenes)
        else:
            n = len(scenes)
            idx = permutation(n)
            split = int(n * self.opts.train_frac)
            train_idxs = idx < split
            test_idxs = idx >= split

        train_dataset, test_dataset = (
            GaDataset(
                df=scenes[idx],
                responses=responses[idx],
                root_dir=self.opts.data_file.parent,
                k_eig=self.k_eig,
                op_cache_dir=op_cache_dir,
                normalize=False,
                file_mode=self.opts.mesh_file_mode,
                weights=weights[idx] if weights is not None else None,
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
        writer = SummaryWriter(log_dir=str(self.log_folder), flush_secs=10)

        return Experiment(
            metadata=self,
            model=model,
            device=device,
            optimizer=optimizer,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            writer=writer,
        )


@dataclass
class Options:
    mesh_file_mode: str
    data_file: Path
    data_dir: Path
    log_folder: Path
    log_file: Path

    input_features: tuple[str]
    channel: tuple[int | Sequence[int]]
    k_eig: tuple[int]
    learning_rate: tuple[float]
    decay_every: tuple[int]
    decay_rate: tuple[float]
    dropout: tuple[bool]
    spike_window: tuple[tuple[float, float]]
    weight_error: tuple[None | Literal['response', 'binned']]
    n_faces: tuple[None | int]

    train_frac: float = 0.95
    n_epoch: int = 1

    @staticmethod
    def for_timestamp(data_file: Path, stamp: str | None = None, **kwargs) -> Options:
        stamp = stamp or datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        data_dir = data_file.parent
        log_folder = data_dir / stamp
        log_folder.mkdir(parents=True, exist_ok=True)

        return Options(
            data_dir=data_dir,
            data_file=data_file,
            log_folder=log_folder,
            log_file=log_folder / f'diffnet_log_{stamp}.txt',
            **kwargs
        )

    def metadata(self, **kwargs) -> Metadata:
        stamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        log_folder = self.log_folder / stamp
        log_folder.mkdir(parents=True, exist_ok=True)
        return Metadata(
            opts=self,
            log_folder=log_folder,
            model_file=log_folder / f'diffnet_model_{stamp}.pt',
            metadata_file=log_folder / f'metadata_{stamp}.pt',
            **kwargs,
        )

    def iter_metadata(self) -> Iterator[Metadata]:
        hparams = {k: v for k, v in self.__dict__.items() if isinstance(v, tuple)}
        for hp in hparam_combinations(hparams):
            yield self.metadata(**hp)

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


@dataclass
class Experiment:
    metadata: Metadata
    model: DiffusionNet
    device: torch.device
    optimizer: torch.optim.Adam
    train_dataset: GaDataset
    test_dataset: GaDataset
    writer: SummaryWriter

    def load_item(self, data):
        verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels, weights = data
        verts = verts.to(self.device)
        faces = faces.to(self.device)
        _frames = frames.to(self.device)
        mass = mass.to(self.device)
        L = L.to(self.device)
        evals = evals.to(self.device)
        evecs = evecs.to(self.device)
        gradX = gradX.to(self.device)
        gradY = gradY.to(self.device)

        if weights is not None:
            if not isinstance(weights, torch.Tensor):
                weights = torch.Tensor(np.asarray(weights))
            weights = weights.to(self.device)

        if not isinstance(labels, torch.Tensor):
            labels = torch.Tensor(labels)

        labels = labels.to(self.device)

        # if augment_random_rotate:
        #     verts = diffusion_net.utils.random_rotate_points(verts)

        # Construct features
        if self.metadata.input_features == 'xyz':
            features = verts
        elif self.metadata.input_features == 'hks':
            features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)
        else:
            raise NotImplementedError(self.metadata.input_features)

        # Apply the model
        preds = self.model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
        return labels, preds, weights

    def train_epoch(self, loader: DataLoader, epoch: int) -> float:
        # Returns mean training loss
        if epoch > 0 and epoch % self.metadata.decay_every == 0:
            self.metadata.learning_rate *= self.metadata.decay_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.metadata.learning_rate

        self.model.train()
        self.optimizer.zero_grad()
        losses = []

        for data in tqdm(loader):
            labels, preds, weights = self.load_item(data)
            err = torch.square(preds - labels)
            if weights is not None:
                err = err * weights

            loss = torch.mean(err)
            losses.append(toNP(loss))
            loss.backward()

            # Step the optimizer
            self.optimizer.step()
            self.optimizer.zero_grad()

        return np.mean(losses)

    def test(self, loader):
        self.model.eval()
        losses = []

        with torch.no_grad():
            for data in tqdm(loader):
                labels, preds, weights = self.load_item(data)
                err = torch.square(preds - labels)
                if weights is not None:
                    err = err * weights

                loss = torch.mean(err)
                losses.append(toNP(loss))

        return np.mean(losses)

    def predict(self, loader, agg_fn=np.concatenate):
        self.model.eval()
        obs, preds = [], []

        with torch.no_grad():
            for data in tqdm(loader):
                _obs, _preds, _weights = self.load_item(data)
                obs.append(_obs.cpu().numpy())
                preds.append(_preds.cpu().numpy())

        return agg_fn(obs), agg_fn(preds)


def main():
    logger = logging.getLogger(__name__)
    opts = Options.for_timestamp(
        # data_file=Path(r"D:\resynth\run_51_52\1k_faces\run00051_resynth.hdf"),
        # channel=(0, 2, 29, 5, 17, 23, 14, 31, 18, 30, 7, 25, 3, 9),
        # data_file=Path(r"D:\resynth\run_48_49\1k_faces\run00048_resynth.hdf"),
        # channel=(14, 17, 29, 23, 2, 0, 13, 31, 3, 26, 28, 9, 20, 11, 18),
        # data_file=Path(r"D:\resynth\run_42_43\1k_faces\run00042_resynth.hdf"),
        # channel=(14, 17, 29, 23, 2, 0, 13, 31, 3, 26, 28, 9, 20, 11, 18),
        # data_file=Path(r"D:\resynth\run_38_39\1k_faces\run00038_resynth.hdf"),
        # channel=(14, 17, 29, 23, 2, 0, 13, 31, 3, 26, 9, 20, 11, 18),
        # data_file=Path(r"D:\resynth\run_20_21\1k_faces\run00020_resynth.hdf"),
        # channel=(2, 17, 13, 29, 14, 7, 23), # , 3, 28, 8, 12, 18, 31, 27, 4),
        # data_file=Path(r"D:\resynth\run_09_10\1k_faces\run00009_resynth.hdf"),
        # channel=(29, 2, 19, 31, 0, 23, 12, 14, 18, 8),

        # data_file=Path(r"D:\resynth\run_48_49\resynth_everything3\run00048_resynth.hdf"),
        data_file=Path(r"D:\resynth\run_48_49\many_faces\run00048_resynth.hdf"),
        n_epoch=250,
        mesh_file_mode='simplified',
        train_frac=0.95,

        channel=((14, 17, 29, 23, 2, 0, 13, 31, 3, 26, 28, 9, 20, 11, 18),),
        input_features=('hks',),
        spike_window=((0.07, 0.75),),
        weight_error=('response',),
        k_eig=(128,),
        learning_rate=(1e-3,),
        decay_every=(50,),
        decay_rate=(0.5,),
        dropout=(True,),
        n_faces=(1500, 1250, 1000, 750, 500),
    )
    opts.init_log()
    train_test_scenes = None
    metas = []

    for meta in opts.iter_metadata():
        metas.append(meta)

        train_dataset, test_dataset = meta.load_datasets(precalc_ops=False, train_test_scenes=train_test_scenes)
        train_test_scenes = train_dataset.df.scene.values, test_dataset.df.scene.values
        exp = meta.experiment(train_dataset=train_dataset, test_dataset=test_dataset)

        train_loader = DataLoader(exp.train_dataset, batch_size=None, shuffle=True)
        test_loader = DataLoader(exp.test_dataset, batch_size=None)

        best_loss = np.inf

        for epoch in range(opts.n_epoch):
            train_loss = exp.train_epoch(train_loader, epoch)
            exp.writer.add_scalar(f'loss/train', train_loss, epoch)
            test_loss = exp.test(test_loader)
            exp.writer.add_scalar(f'loss/test', test_loss, epoch)
            logger.debug(f"Epoch {epoch}: Train: {train_loss:.5e}  Test: {test_loss:.5e}")

            if test_loss < best_loss:
                best_loss = test_loss
                logger.debug('Saving best test loss to %s', meta.model_file)
                torch.save(exp.model.state_dict(), meta.model_file)

        mf = meta.model_file.with_suffix('.last' + meta.model_file.suffix)
        logger.debug("Saving last model to %s", mf)
        torch.save(exp.model.state_dict(), meta.model_file)

    metadata = dict(
        opts=opts,
        metadata=metas,
        train_scenes=train_test_scenes[0],
        test_scenes=train_test_scenes[1],
    )
    final = opts.log_folder / 'opts_and_metadata.pt'
    torch.save(metadata, final)
    logger.debug("Saving all metadata to %s", final)


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

    def metadata(self):
        metafile = self.folder / f"metadata_{self.folder.parts[-1]}.pt"
        return torch.load(metafile)
