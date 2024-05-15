import os
import sys
import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net
from shrec11_dataset import Shrec11MeshDataset_Simplified, Shrec11MeshDataset_Original


@dataclass
class Options:
    device: Any
    n_epoch: int
    lr: float
    input_features: str
    augment_random_rotate: bool
    k_eig: int = 128
    decay_every: int = 50
    decay_rate: float = 0.5
    label_smoothing_fac: float = 0.2
    base_path: Path = Path(__file__).parent

    def load_datasets(self, dataset_type: str, split_size: int):
        op_cache_dir = self.base_path / "data" / "op_cache"
        if dataset_type == "simplified":
            dataset_path = self.base_path / "data" / "simplified"
            train_dataset = Shrec11MeshDataset_Simplified(
                dataset_path, split_size=split_size, k_eig=self.k_eig, op_cache_dir=op_cache_dir)

            test_dataset = Shrec11MeshDataset_Simplified(
                dataset_path, split_size=None, k_eig=self.k_eig, op_cache_dir=op_cache_dir,
                exclude_dict=train_dataset.entries)

        elif dataset_type == "original":
            dataset_path = self.base_path / "data" / "original"
            train_dataset = Shrec11MeshDataset_Original(
                dataset_path, split_size=split_size, k_eig=self.k_eig, op_cache_dir=op_cache_dir)

            test_dataset = Shrec11MeshDataset_Original(
                dataset_path, split_size=None, k_eig=self.k_eig, op_cache_dir=op_cache_dir,
                exclude_dict=train_dataset.entries)
        else:
            raise ValueError("Unrecognized dataset type")

        train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=None)
        return train_loader, test_loader

    def build_model(self, C_out: int):
        C_in = 3 if self.input_features == 'xyz' else 16
        return diffusion_net.layers.DiffusionNet(
            C_in=C_in, C_out=C_out, C_width=64, N_block=4,
            last_activation=lambda x: torch.nn.functional.log_softmax(x, dim=-1),
            outputs_at='global_mean',
            dropout=False,
        )


def train_epoch(epoch, optimizer, model, loader, opts: Options):
    # Implement lr decay
    if epoch > 0 and epoch % opts.decay_every == 0:
        opts.lr *= opts.decay_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = opts.lr

    model.train()
    optimizer.zero_grad()

    correct = 0
    total_num = 0
    for data in tqdm(loader):

        # Get data
        verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels = data

        # Move to device
        verts = verts.to(opts.device)
        faces = faces.to(opts.device)
        frames = frames.to(opts.device)
        mass = mass.to(opts.device)
        L = L.to(opts.device)
        evals = evals.to(opts.device)
        evecs = evecs.to(opts.device)
        gradX = gradX.to(opts.device)
        gradY = gradY.to(opts.device)
        labels = labels.to(opts.device)

        # Randomly rotate positions
        if opts.augment_random_rotate:
            verts = diffusion_net.utils.random_rotate_points(verts)

        # Construct features
        if opts.input_features == 'xyz':
            features = verts
        elif opts.input_features == 'hks':
            features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)
        else:
            raise ValueError(opts.input_features)

        # Apply the model
        preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)

        # Evaluate loss
        loss = diffusion_net.utils.label_smoothing_log_loss(preds, labels, opts.label_smoothing_fac)
        loss.backward()

        # track accuracy
        pred_labels = torch.max(preds, dim=-1).indices
        this_correct = pred_labels.eq(labels).sum().item()
        correct += this_correct
        total_num += 1

        # Step the optimizer
        optimizer.step()
        optimizer.zero_grad()

    train_acc = correct / total_num
    return train_acc


def test(model, loader, opts: Options):

    model.eval()

    correct = 0
    total_num = 0
    with torch.no_grad():

        for data in tqdm(loader):

            # Get data
            verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels = data

            # Move to opts.device
            verts = verts.to(opts.device)
            faces = faces.to(opts.device)
            frames = frames.to(opts.device)
            mass = mass.to(opts.device)
            L = L.to(opts.device)
            evals = evals.to(opts.device)
            evecs = evecs.to(opts.device)
            gradX = gradX.to(opts.device)
            gradY = gradY.to(opts.device)
            labels = labels.to(opts.device)

            # Construct features
            if opts.input_features == 'xyz':
                features = verts
            elif opts.input_features == 'hks':
                features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)
            else:
                raise ValueError(opts.input_features)

            # Apply the model
            preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)

            # track accuracy
            pred_labels = torch.max(preds, dim=-1).indices
            this_correct = pred_labels.eq(labels).sum().item()
            correct += this_correct
            total_num += 1

    test_acc = correct / total_num
    return test_acc


def experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_features", type=str,
                        help="what features to use as input ('xyz' or 'hks') default: hks", default='hks')
    parser.add_argument("--dataset_type", type=str,
                        help="which variant of the dataset to use ('original', or 'simplified') default: original",
                        default='original')
    parser.add_argument("--split_size", type=int, help="how large of a training set per-class default: 10",
                        default=10)
    parser.add_argument("--n_epoch", type=int, help="number of epochs default: 200",
                        default=200)
    args = parser.parse_args()

    opts = Options(
        device=torch.device('cuda:0'),
        input_features=args.input_features,
        k_eig=128,
        n_epoch=50,
        lr=1e-3,
        decay_every=50,
        decay_rate=0.5,
        augment_random_rotate=args.input_features == 'xyz',
        label_smoothing_fac=0.2,
    )

    train_loader, test_loader = opts.load_datasets(dataset_type=args.dataset_type, split_size=args.split_size)
    n_class = 30
    model = opts.build_model(C_out=n_class)
    model = model.to(opts.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
    model = model.to(opts.device)

    print("Training...")
    for epoch in range(args.n_epoch):
        train_acc = train_epoch(epoch=epoch, optimizer=optimizer, model=model, loader=train_loader, opts=opts)
        test_acc = test(model=model, loader=test_loader, opts=opts)
        print(f"Epoch {epoch} - Train overall: {100 * train_acc:06.3f}%  Test overall: {100 * test_acc:06.3f}")

    test_acc = test(model=model, loader=test_loader, opts=opts)
    print(f"Overall test accuracy: {100 * test_acc:06.3f}%")

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_save_path = str(opts.base_path / f'trained_{args.dataset_type}_{opts.input_features}_{timestamp}.pth')
    print("Saving last model to " + model_save_path)
    torch.save(model.state_dict(), model_save_path)


if __name__ == '__main__':
    experiment()