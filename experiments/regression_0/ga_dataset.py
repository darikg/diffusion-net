import os
import sys
from pathlib import Path
from typing import cast, Literal, List, Tuple

import potpourri3d as pp3d
import torch
from pandas import DataFrame
from pandas import read_hdf
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net


class GaDataset(Dataset):
    def __init__(self, df: DataFrame, root_dir, k_eig, op_cache_dir=None, normalize=False):
        self.df = df
        self.root_dir = root_dir
        self.k_eig = k_eig
        self.op_cache_dir = op_cache_dir
        self.entries = {}
        self.normalize = normalize

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

        if self.normalize:
            verts = diffusion_net.geometry.normalize_positions(verts)

        faces = torch.tensor(faces)
        frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.get_operators(
            verts, faces, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)
        return verts, faces, frames, mass, L, evals, evecs, gradX, gradY, response

    @staticmethod
    def load_data(
            data_file: Path,
            channel: int,
            file_mode: str,
            norm_responses: bool,
            norm_verts: bool,
            k_eig: int,
    ) -> Tuple[DataFrame, Path]:
        scenes = cast(DataFrame, read_hdf(data_file, 'scenes')).reset_index()
        scenes = scenes[scenes[file_mode] != '']
        assert isinstance(channel, int)
        responses = cast(DataFrame, read_hdf(data_file, 'responses')).reset_index()
        responses = responses[responses['channel'] == channel].set_index('scene')
        if norm_responses:
            r0, r1 = responses.min(), responses.max()
            responses = (responses - r0) / (r1 - r0)

        scenes = scenes.join(responses, on='scene', how='inner')
        op_cache_dir = data_file.parent / 'op_cache'

        print('Pre-calculating operators')
        for mesh_file in tqdm(scenes[file_mode]):
            verts, faces = pp3d.read_mesh(str(data_file.parent / mesh_file))
            verts = torch.tensor(verts).float()
            faces = torch.tensor(faces)
            if norm_verts:
                verts = diffusion_net.geometry.normalize_positions(verts)

            _ = diffusion_net.geometry.get_operators(verts, faces, k_eig=k_eig, op_cache_dir=op_cache_dir)

        return scenes, op_cache_dir
