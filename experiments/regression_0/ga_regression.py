from __future__ import annotations
import logging
import os
import sys
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net
from diffusion_net.utils import toNP
from ga_dataset import GaDataset, make_model


@dataclass
class Options:
    input_features: str
    data_file: Path
    data_dir: Path
    log_file: Path
    channel: int
    k_eig: int = 128
    learning_rate: float = 1e-3
    decay_every = 50
    decay_rate = 0.5
    augment_random_rotate = False



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
        args = parser.parse_args()
        data_file = Path(args.file)
        data_dir = data_file.parent
        log_file = data_dir / 'log.txt'
        if log_file.exists():
            log_file.unlink()

        return Options(
            data_file=data_file,
            data_dir=data_dir,
            log_file=log_file,
            channel=args.channel,
            input_features=args.input_features,

        )

    def init_log(self):
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )

        # set up logging to console
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    def load_datasets(self) -> Tuple[GaDataset, GaDataset]:
        train_dataset, test_dataset = GaDataset.load_lineages(
            data_file=self.data_file, k_eig=self.k_eig, channel=self.channel)

    def make_model(self, C_in):
        return diffusion_net.layers.DiffusionNet(
            C_in=C_in,
            C_out=1,
            C_width=64,
            N_block=4,
            last_activation=None,
            outputs_at='global_mean',
            dropout=False,
        )



def train_epoch(epoch):

    # Implement lr decay
    if epoch > 0 and epoch % decay_every == 0:
        global lr 
        lr *= decay_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr 

    # Set model to 'train' mode
    model.train()
    optimizer.zero_grad()
    
    losses = []

    for data in tqdm(train_loader):

        # Get data
        verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels = data

        # Move to device
        verts = verts.to(device)
        faces = faces.to(device)
        frames = frames.to(device)
        mass = mass.to(device)
        L = L.to(device)
        evals = evals.to(device)
        evecs = evecs.to(device)
        gradX = gradX.to(device)
        gradY = gradY.to(device)
        labels = labels.to(device)
        
        # Randomly rotate positions
        if augment_random_rotate:
            verts = diffusion_net.utils.random_rotate_points(verts)

        # Construct features
        if input_features == 'xyz':
            features = verts
        elif input_features == 'hks':
            features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)

        # Apply the model
        preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)

        # Evaluate loss
        # loss = diffusion_net.utils.label_smoothing_log_loss(preds, labels, label_smoothing_fac)
        loss = torch.mean(torch.square(preds - labels))
        losses.append(toNP(loss))
        loss.backward()

        # Step the optimizer
        optimizer.step()
        optimizer.zero_grad()

    train_loss = np.mean(losses)
    return train_loss


# Do an evaluation pass on the test dataset 
def test():
    
    model.eval()
    losses = []

    with torch.no_grad():
    
        for data in tqdm(test_loader):

            # Get data
            verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels = data

            # Move to device
            verts = verts.to(device)
            faces = faces.to(device)
            frames = frames.to(device)
            mass = mass.to(device)
            L = L.to(device)
            evals = evals.to(device)
            evecs = evecs.to(device)
            gradX = gradX.to(device)
            gradY = gradY.to(device)
            labels = labels.to(device)
            
            # Construct features
            if input_features == 'xyz':
                features = verts
            elif input_features == 'hks':
                features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)

            # Apply the model
            preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)

            loss = torch.mean(torch.square(preds - labels))
            losses.append(toNP(loss))

    mean_loss = np.mean(losses)
    return mean_loss


def main():
    logger = logging.getLogger(__name__)
    device = torch.device('cuda:0')
    opts = Options.parse()
    train_dataset, test_dataset = opts.load_datasets()
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=None)

    model =
    model = model.to(device)

    # === Optimize
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print("Training...")
for epoch in range(args.n_epoch):
    train_loss = train_epoch(epoch)
    test_loss = test()
    print("Epoch {} - Train overall: {:.5e}  Test overall: {:.5e}".format(epoch, train_loss, test_loss))


model_save_path = str(data_dir / 'trained.pth')
print(" ==> saving last model to " + model_save_path)
torch.save(model.state_dict(), model_save_path)

