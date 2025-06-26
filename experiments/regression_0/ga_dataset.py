import os
import sys
from pathlib import Path
from typing import cast, Tuple, Sequence

import numpy as np
import potpourri3d as pp3d
import torch
from pandas import DataFrame
from pandas import read_hdf
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net  # noqa


class GaDataset(Dataset):
    def __init__(
            self,
            df: DataFrame,
            responses: np.ndarray,
            root_dir,
            k_eig,
            file_mode: str,
            op_cache_dir=None,
            normalize=False,
    ):
        self.df = df
        self.responses = responses
        self.root_dir = root_dir
        self.k_eig = k_eig
        self.op_cache_dir = op_cache_dir
        self.entries = {}
        self.normalize = normalize
        self.file_mode = file_mode

        # center and unit scale
        # verts = diffusion_net.geometry.normalize_positions(verts)

        # for ind, label in enumerate(self.labels_list):
        #     self.labels_list[ind] = torch.tensor(label)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        response = self.responses[idx]
        path = self.df[self.file_mode].iloc[idx]
        mesh_file = self.root_dir / path

        if mesh_file.suffix == '.vtp':
            import pyvista as pv
            mesh = pv.read(mesh_file)
            verts, faces = mesh.points, mesh.regular_faces
        else:
            verts, faces = pp3d.read_mesh(str(mesh_file))

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
            channel: int | Sequence[int],
            file_mode: str,
            norm_verts: bool,
            k_eig: int,
            spike_window: tuple[float, float],
            precalc_ops: bool = True,
    ) -> Tuple[DataFrame, np.ndarray, Path]:
        scenes = cast(DataFrame, read_hdf(data_file, 'scenes'))
        scenes = scenes[scenes[file_mode] != '']

        t0, t1 = spike_window
        spikes = cast(DataFrame, read_hdf(data_file, 'spikes'))
        spk_rates = (
            spikes.time.between(t0, t1, inclusive='left')
            .groupby(['stim_id', 'task_id', 'channel'])
            .sum()
            .groupby(['stim_id', 'channel'])
            .median()
        ) / (t1 - t0)
        responses = spk_rates.rename('response').unstack('channel')

        r0, r1 = responses.min(axis=0), responses.max(axis=0)
        responses_ = ((responses - r0) / (r1 - r0)).replace((-np.inf, np.inf, np.nan), 0)
        scenes_ = scenes[scenes.index.isin(responses_.index)].reset_index()
        responses_ = responses_.loc[(scenes.index, channel)].values  # (n_scenes * n_channels)
        op_cache_dir = data_file.parent / 'op_cache'

        if precalc_ops:
            print('Pre-calculating operators')
            for f in tqdm(scenes[file_mode]):
                mesh_file = data_file.parent / f
                if mesh_file.suffix == '.ply':
                    verts, faces = pp3d.read_mesh(str(mesh_file))
                else:
                    import pyvista as pv
                    mesh = pv.read(mesh_file)
                    verts, faces = mesh.points, mesh.regular_faces

                verts = torch.tensor(verts).float()
                faces = torch.tensor(faces)
                if norm_verts:
                    verts = diffusion_net.geometry.normalize_positions(verts)

                _ = diffusion_net.geometry.get_operators(verts, faces, k_eig=k_eig, op_cache_dir=op_cache_dir)

        return scenes_, responses_, op_cache_dir
