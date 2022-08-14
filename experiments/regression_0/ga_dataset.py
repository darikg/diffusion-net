import shutil
import os
import sys
import random
from pathlib import Path
from typing import Optional

import numpy as np

import torch
from pandas import DataFrame, read_parquet
from torch.utils.data import Dataset

import potpourri3d as pp3d

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net
from diffusion_net.utils import toNP



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

        # Precompute operators
        # self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list = diffusion_net.geometry.get_all_operators(self.verts_list, self.faces_list, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        response = self.df.response.values[idx]
        path = self.df.iloc[idx].mesh
        verts, faces = pp3d.read_mesh(str(self.root_dir / path))
        verts = torch.tensor(verts).float()
        faces = torch.tensor(faces)
        frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.get_operators(
            verts, faces, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)
        return verts, faces, frames, mass, L, evals, evecs, gradX, gradY, response

    @staticmethod
    def load_lineages(root_dir: Path, k_eig, op_cache_dir=None):
        scenes = read_parquet(root_dir / 'scenes.parquet')
        scenes = scenes.loc[~scenes.missing_files]
        scenes = scenes[['scene', 'lineage', 'mesh']]

        responses = read_parquet(root_dir / 'responses.parquet')
        responses = responses.groupby('scene').response.max()

        scenes = scenes.join(responses, on='scene', how='inner').set_index('scene')

        for mesh_file in scenes.mesh:
            print(str(mesh_file))
            verts, faces = pp3d.read_mesh(str(root_dir / mesh_file))
            verts = torch.tensor(verts).float()
            faces = torch.tensor(faces)
            frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.get_operators(
                verts, faces, k_eig=k_eig, op_cache_dir=op_cache_dir)

        for _, df in scenes.groupby('lineage'):
            yield GaDataset(df=df, root_dir=root_dir, k_eig=k_eig, op_cache_dir=op_cache_dir)

