import os
import sys
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net
from diffusion_net.utils import toNP
from ga_dataset import GaDataset


# === Options

# Parse a few args
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
args = parser.parse_args()

# system things
device = torch.device('cuda:0')
dtype = torch.float32

# model
input_features = args.input_features  # one of ['xyz', 'hks']
k_eig = 128

# training settings
n_epoch = 200
lr = 1e-3
decay_every = 50
decay_rate = 0.5
# augment_random_rotate = (input_features == 'xyz')
augment_random_rotate = False
label_smoothing_fac = 0.2


# Important paths
data_file = Path(args.file)
data_dir = data_file.parent
op_cache_dir = data_dir / "op_cache"
op_cache_dir.mkdir(exist_ok=True)

# === Load datasets
train_dataset, test_dataset = GaDataset.load_lineages(
    data_file=data_file, k_eig=k_eig, op_cache_dir=op_cache_dir, channel=args.channel)

train_loader = DataLoader(train_dataset,  batch_size=None, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=None)


# === Create the model
C_in = {'xyz': 3, 'hks': 16}[input_features]  # dimension of input features

model = diffusion_net.layers.DiffusionNet(
    C_in=C_in,
    C_out=1,
    C_width=64,
    N_block=4,
    # last_activation=lambda x : torch.nn.functional.log_softmax(x,dim=-1),
    outputs_at='global_mean',
    dropout=False,
)


model = model.to(device)

# === Optimize
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


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


print("Training...")

for epoch in range(n_epoch):
    train_loss = train_epoch(epoch)
    test_loss = test()
    print("Epoch {} - Train overall: {:.5e}  Test overall: {:.5e}".format(epoch, train_loss, test_loss))

