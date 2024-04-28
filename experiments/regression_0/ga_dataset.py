import os
import sys
from pathlib import Path
from typing import cast

import potpourri3d as pp3d
import torch
from pandas import DataFrame
from pandas import read_hdf
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net


class GaDataset(Dataset):
    def __init__(self, df: DataFrame, root_dir, k_eig, op_cache_dir=None):
        self.df = df
        self.root_dir = root_dir
        self.k_eig = k_eig
        self.op_cache_dir = op_cache_dir
        self.entries = {}

        # center and unit scale
        # verts = diffusion_net.geometry.normalize_positions(verts)

        # for ind, label in enumerate(self.labels_list):
        #     self.labels_list[ind] = torch.tensor(label)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        response = self.df.response.values[idx]
        path = self.df.iloc[idx].simplified
        verts, faces = pp3d.read_mesh(str(self.root_dir / path))
        verts = torch.tensor(verts).float()
        faces = torch.tensor(faces)
        frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.get_operators(
            verts, faces, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)
        return verts, faces, frames, mass, L, evals, evecs, gradX, gradY, response

    @staticmethod
    def load_lineages(data_file: Path, k_eig, channel: int, op_cache_dir=None):
        scenes = cast(DataFrame, read_hdf(data_file, 'scenes')).reset_index()
        scenes = scenes[scenes['simplified'] != '']
        assert isinstance(channel, int)
        responses = cast(DataFrame, read_hdf(data_file, 'responses')).reset_index()
        responses = responses[responses['channel'] == channel].set_index('scene')
        scenes = scenes.join(responses, on='scene', how='inner')

        print('Pre-calculating operators')
        for mesh_file in tqdm(scenes.simplified):
            # print(str(mesh_file))
            verts, faces = pp3d.read_mesh(str(data_file.parent / mesh_file))
            verts = torch.tensor(verts).float()
            faces = torch.tensor(faces)
            frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.get_operators(
                verts, faces, k_eig=k_eig, op_cache_dir=op_cache_dir)

        for _, df in scenes.groupby('lineage'):
            yield GaDataset(df=df, root_dir=data_file.parent, k_eig=k_eig, op_cache_dir=op_cache_dir)

