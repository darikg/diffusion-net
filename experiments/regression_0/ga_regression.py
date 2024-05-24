from __future__ import annotations
import logging
import os
import sys
import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple, Any

import torch
from numpy import array
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net
from diffusion_net.utils import toNP
from diffusion_net.layers import DiffusionNet
from ga_dataset import GaDataset


@dataclass
class Options:
    input_features: str
    data_file: Path
    data_dir: Path
    mesh_file_mode: str
    log_file: Path
    model_file: Path
    n_epoch: int
    channel: int
    k_eig: int = 128
    learning_rate: float = 1e-3
    decay_every = 50
    decay_rate = 0.5
    dropout: bool = False
    augment_random_rotate = False
    norm_verts: bool = False
    norm_response: bool = False
    args: Any = None

    @staticmethod
    def for_timestamp(stamp: str, data_file: Path, **kwargs) -> Options:
        data_dir = data_file.parent

        log_file = data_dir / f'log_{stamp}.txt'
        model_file = data_dir / f'model_{stamp}.pth'
        return Options(**kwargs, log_file=log_file, model_file=model_file, data_dir=data_dir, data_file=data_file)

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
        args = parser.parse_args()
        stamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        return Options.for_timestamp(
            stamp=stamp,
            data_file=Path(args.file),
            channel=args.channel,
            n_epoch=args.n_epoch,
            input_features=args.input_features,
            dropout=args.dropout,
            mesh_file_mode=args.file_mode,
            norm_verts=args.norm_verts,
            norm_response=args.norm_response,
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

    def load_datasets(self, lineage_groups: list[list[int]]) -> Tuple[GaDataset, GaDataset]:
        train_dataset, test_dataset = GaDataset.load_lineages(
            data_file=self.data_file,
            k_eig=self.k_eig,
            channel=self.channel,
            lineage_groups=lineage_groups,
            file_mode=self.mesh_file_mode,
            norm_verts=self.norm_verts,
            norm_responses=self.norm_response,
        )
        return train_dataset, test_dataset

    def make_model(self) -> DiffusionNet:
        return DiffusionNet(
            C_in=3 if self.input_features == 'xyz' else 16,
            C_out=1,
            C_width=64,
            N_block=4,
            last_activation=None,
            outputs_at='global_mean',
            dropout=self.dropout,
        )

    def experiment(self) -> Experiment:
        device = torch.device('cuda:0')
        train_dataset, test_dataset = self.load_datasets(lineage_groups=[[1, 2], [0]])
        model = self.make_model()
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        return Experiment(
            opts=self,
            model=model,
            device=device,
            optimizer=optimizer,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
        )


@dataclass
class Experiment:
    opts: Options
    model: DiffusionNet
    device: torch.device
    optimizer: torch.optim.Adam
    train_dataset: GaDataset
    test_dataset: GaDataset

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

        # Set model to 'train' mode
        self.model.train()
        self.optimizer.zero_grad()
        losses = []

        for data in tqdm(loader):
            labels, preds = self.load_item(data)

            # Evaluate loss
            # loss = diffusion_net.utils.label_smoothing_log_loss(preds, labels, label_smoothing_fac)
            loss = torch.mean(torch.square(preds - labels))
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
                obs.append(_obs.item())
                preds.append(_preds.item())

        return array(obs), array(preds)


def main():
    logger = logging.getLogger(__name__)
    opts = Options.parse()
    opts.init_log()

    exp = opts.experiment()
    train_loader = DataLoader(exp.train_dataset, batch_size=None, shuffle=True)
    test_loader = DataLoader(exp.test_dataset, batch_size=None)

    for epoch in range(opts.n_epoch):
        train_loss = exp.train_epoch(train_loader, epoch)
        test_loss = exp.test(test_loader)
        logger.debug(f"Epoch {epoch}: Train: {train_loss:.5e}  Test: {test_loss:.5e}")

    logger.debug("Saving last model to %s", opts.model_file)
    torch.save(exp.model.state_dict(), opts.model_file)


if __name__ == '__main__':
    main()
