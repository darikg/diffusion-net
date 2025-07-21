import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast, Tuple, Sequence, Literal, Callable, TypeAlias

import numpy as np
import pandas as pd
import potpourri3d as pp3d
import torch
from pandas import DataFrame, Series
from pandas import read_hdf
from torch.utils.data import Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net  # noqa


def calc_binned_weights(r: pd.Series, n_bins=10, plot=False):
    import scipy.optimize
    bins = pd.qcut(r, n_bins, duplicates='drop')
    binned = r.groupby(bins, observed=True).aggregate(n='count', avg='mean')
    binned['frac'] = binned.n / len(r)
    binned['weight'] = 1 / binned.frac * binned.avg

    b0, b1 = binned.avg.min(), binned.avg.max()
    weight0 = pd.Series(binned.weight[bins].values, index=bins.index)  # (n_stim,) Constant in each bin
    weight1 = weight0 * (b0 + (b1 - b0) * r)  # Linear in each bin

    def exp_fn(t, a_, b_):
        return a_ * np.exp(b_ * t)

    (a, b), _ = scipy.optimize.curve_fit(exp_fn, r, weight1)  # noqa
    weight2 = exp_fn(r, a, b)

    if plot:
        from matplotlib import pyplot as plt
        plt.plot(r, r, 'k.', r, weight0, 'b.', r, weight1, 'g.', r, weight2, 'c.')

    weight_fn = lambda t: exp_fn(t, a, b)  # noqa
    return weight2, weight_fn


def fit_channel_weights(responses: pd.DataFrame) -> tuple[pd.DataFrame, list[Callable[[Series], Series]]]:
    ch_weights, ch_fits = [], []

    for ch, r in responses.items():
        w, fit_fn = calc_binned_weights(responses.loc[:, ch])
        ch_weights.append(w.rename(ch))
        ch_fits.append(fit_fn)

    weights = pd.concat(ch_weights, axis=1)
    return weights, ch_fits


def apply_fit_fns(responses: pd.DataFrame, fit_fns: list[Callable[[Series], Series]]):
    ch_weights = [
        fit_fn(r).rename(ch)
        for (ch, r), fit_fn in zip(responses.items(), fit_fns)
    ]
    return pd.concat(ch_weights, axis=1)


WeightErrorMode = Literal['response', 'binned']
UseVisibleMode = Literal['orig', 'shuffled']
UseColorMode = Literal['orig', 'shuffled']


_NormMethod = Literal['mean', 'bbox']
_ScaleMethod = Literal['max_rad', 'area']
NormVertMode: TypeAlias = tuple[_NormMethod, _ScaleMethod]


@dataclass
class MeshData:
    verts: torch.Tensor
    faces: torch.Tensor
    frames: torch.Tensor
    mass: torch.Tensor
    L: torch.Tensor
    evals: torch.Tensor
    evecs: torch.Tensor
    gradX: torch.Tensor
    gradY: torch.Tensor
    labels: torch.Tensor
    weight: torch.Tensor | None
    visible: torch.Tensor | None
    color: torch.Tensor | None


class GaDataset(Dataset):
    def __init__(
            self,
            df: DataFrame,
            responses: np.ndarray,
            root_dir,
            k_eig,
            file_mode: str,
            use_visible: UseVisibleMode | None,
            use_color: UseColorMode | None,
            norm_verts: NormVertMode | None,
            op_cache_dir=None,
            weights: np.ndarray | None = None,

    ):
        self.df = df
        self.responses = responses
        self.weights = weights
        self.root_dir = root_dir
        self.k_eig = k_eig
        self.op_cache_dir = op_cache_dir
        self.entries = {}
        self.file_mode = file_mode
        self.use_visible = use_visible
        self.use_color = use_color
        self.norm_verts = norm_verts

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx) -> MeshData:
        response = self.responses[idx]
        weight = self.weights[idx] if self.weights is not None else None
        path = self.df[self.file_mode].iloc[idx]
        mesh_file = self.root_dir / path
        visible = color = None

        if mesh_file.suffix == '.vtp':
            import pyvista as pv
            mesh = pv.read(mesh_file)
            verts, faces = mesh.points, mesh.regular_faces

            if self.use_visible:
                if self.use_visible == 'orig':
                    visible = torch.tensor(mesh.point_data['visible']).float()
                elif self.use_visible == 'shuffled':
                    visible = torch.tensor(mesh.point_data['shuffled_visible']).float()
                else:
                    raise ValueError(self.use_visible)

            if self.use_color:
                if self.use_color == 'orig':
                    color = torch.tensor(mesh.point_data['color']).float()
                elif self.use_color == 'shuffled':
                    color = torch.tensor(mesh.point_data['shuffled_color']).float()
        else:
            verts, faces = pp3d.read_mesh(str(mesh_file))
            if self.use_visible:
                raise ValueError("Can't get visible from ply files")
            if self.use_color:
                raise ValueError("Can't get color from ply files")

        verts = torch.tensor(verts).float()

        if self.norm_verts:
            verts = diffusion_net.geometry.normalize_positions(
                verts, faces=faces, method=self.norm_verts[0], scale_method=self.norm_verts[1])

        faces = torch.tensor(faces)
        frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.get_operators(
            verts, faces, k_eig=self.k_eig, op_cache_dir=self.op_cache_dir)

        if self.use_visible and visible is None:
            raise ValueError("No mesh visibility data!")

        return MeshData(
            verts=verts,
            faces=faces,
            frames=frames,
            mass=mass,
            L=L,
            evals=evals,
            evecs=evecs,
            gradX=gradX,
            gradY=gradY,
            weight=weight,
            labels=response,
            visible=visible,
            color=color,
        )

    @staticmethod
    def load_data(
            data_file: Path,
            channel: int | Sequence[int],
            file_mode: str,
            spike_window: tuple[float, float],
            weight_error: None | Literal['response', 'binned'] = None,
            n_faces: int | None = None,
    ) -> Tuple[DataFrame, np.ndarray, Path, np.ndarray, list[Callable[[Series], Series]] | None]:
        scenes = cast(DataFrame, read_hdf(data_file, 'scenes'))
        scenes = scenes[scenes[file_mode] != '']

        if n_faces is not None:
            def _map_file(f: str):
                f = Path(f)
                return str(f.with_suffix(f'.{n_faces}_faces' + f.suffix))

            scenes[file_mode] = scenes[file_mode].apply(_map_file)

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
        responses_ = responses_.loc[(scenes.index, channel)]  # .values  # (n_scenes * n_channels)

        if weight_error:
            if weight_error == 'binned':
                weights, fit_fns = fit_channel_weights(responses_)
                weights = weights.values
            elif weight_error == 'response':
                weights = responses_.values
                fit_fns = None
            else:
                raise ValueError(weight_error)
        else:
            weights = fit_fns = None

        op_cache_dir = data_file.parent / 'op_cache'
        return scenes_, responses_.values, op_cache_dir, weights, fit_fns
