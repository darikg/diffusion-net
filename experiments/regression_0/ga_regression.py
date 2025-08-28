from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from functools import cached_property, lru_cache
from itertools import product
from pathlib import Path
from typing import Tuple, Any, Sequence, Iterator, cast, Literal, NamedTuple

import numpy as np
import pandas as pd
import torch
from numpy.random import permutation
from pandas import Series
from tbparse import SummaryReader
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net  # noqa
from diffusion_net.utils import toNP   # noqa
from diffusion_net.layers import DiffusionNet   # noqa
from ga_dataset import GaDataset, UseVisibleMode, WeightErrorMode, MeshData, UseColorMode, NormVertMode, \
    FeatureMode, AugmentMode  # noqa


def hparam_combinations(hparams: dict[str, Sequence[Any]]) -> Iterator[dict[str, Any]]:
    for vals in product(*hparams.values()):
        yield {k: v for k, v in zip(hparams.keys(), vals)}


class Sentinel:
    pass

        
@dataclass
class Metadata:
    opts: Options
    log_folder: Path
    model_file: Path
    metadata_file: Path

    input_features: FeatureMode
    channel: int | Sequence[int]
    k_eig: int = 128
    learning_rate: float = 1e-3
    decay_every: int = 50
    decay_rate: float = 0.5
    n_blocks: int = 4
    dropout: bool = False
    n_faces: int | None = None
    spike_window: tuple[float, float] = (0.07, 0.3)
    isolate_channel_idx: int | None = None
    weight_error: WeightErrorMode | None = None
    augment: AugmentMode | None = None
    use_visible: UseVisibleMode | None = None
    use_color: UseColorMode | None = None
    norm_verts: NormVertMode | None = None
    curr_learning_rate: float | None = None
    ultimate_linear: bool = False

    def __post_init__(self):
        self.curr_learning_rate = self.learning_rate

    def load_data(self, weights: WeightErrorMode | None):
        scenes, responses, weights, fit_fns = GaDataset.load_data(
            data_file=self.opts.data_file,
            channel=self.channel,
            file_mode=self.opts.mesh_file_mode,
            spike_window=self.spike_window,
            weight_error=weights or self.weight_error,
            n_faces=self.n_faces,
            features=self.input_features,
        )
        return scenes, responses, weights, fit_fns

    def load_dataset(
            self,
            weights: WeightErrorMode | None,
            augment: AugmentMode | None | Sentinel = Sentinel,
    ) -> GaDataset:
        scenes, responses, weights, fit_fns = self.load_data(weights=weights)

        if augment is Sentinel:
            augment = self.augment

        dataset = GaDataset(
            df=scenes,
            responses=responses,
            root_dir=self.opts.data_file.parent,
            k_eig=self.k_eig,
            op_cache_dir=self.opts.data_dir / 'op_cache',
            file_mode=self.opts.mesh_file_mode,
            weights=weights,
            use_visible=self.use_visible,
            use_color=self.use_color,
            norm_verts=self.norm_verts,
            features=self.input_features,
            augment=augment,
        )
        return dataset

    def load_datasets(
            self,
            train_test_scenes: tuple[Sequence[int], Sequence[int]] | None = None,
            weight_mode: WeightErrorMode | None = None,
            augment=Sentinel,
    ) -> Tuple[GaDataset, GaDataset]:
        scenes, responses, weights, fit_fns = self.load_data(weights=weight_mode)

        if (ch_idx := self.isolate_channel_idx) is not None:
            if weights is None:
                weights = np.ones(responses.shape)
            n_ch = len(self.channel)
            weights[:, np.arange(n_ch) != ch_idx] = 0

        if augment is Sentinel:
            augment = self.augment

        if train_test_scenes:
            train_scenes, test_scenes = train_test_scenes
            train_idxs = np.isin(scenes.index, train_scenes)
            test_idxs = np.isin(scenes.index, test_scenes)
        else:
            n = len(scenes)
            idx = permutation(n)
            split = int(n * self.opts.train_frac)
            train_idxs = idx < split
            test_idxs = idx >= split

        train_dataset, test_dataset = (
            GaDataset(
                df=scenes[idx],  # type: ignore
                responses=responses[idx],  # type: ignore
                root_dir=self.opts.data_file.parent,
                k_eig=self.k_eig,
                op_cache_dir=self.opts.data_dir / 'op_cache',
                file_mode=self.opts.mesh_file_mode,
                weights=weights[idx] if weights is not None else None,
                use_visible=self.use_visible,
                use_color=self.use_color,
                norm_verts=self.norm_verts,
                features=self.input_features,
                augment=augment,
            )
            for idx in (train_idxs, test_idxs)
        )

        return train_dataset, test_dataset

    def make_model(self, n_channels_out: int = 1) -> DiffusionNet:
        C_in = 3 if self.input_features == 'xyz' else 16
        if self.use_visible:
            C_in += 1
        if self.use_color:
            C_in += 4

        if self.ultimate_linear:
            last_activation = nn.Linear(in_features=n_channels_out, out_features=n_channels_out)
        else:
            last_activation = None

        return DiffusionNet(
            C_in=C_in,
            C_out=n_channels_out,
            C_width=64,
            N_block=self.n_blocks,
            last_activation=last_activation,
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

    def as_series(self) -> pd.Series:
        return pd.Series({k: v for k, v in self.__dict__.items() if k not in ('opts',)})

    @staticmethod
    def stamped(opts: Options, idx: int, mkdir: bool, **kwargs):
        log_folder = opts.log_folder / f'expt_{idx:03}'
        if mkdir:
            log_folder.mkdir(parents=True, exist_ok=True)

        return Metadata(
            # NB DOUBLE CHECK metadata.restamp
            opts=opts,
            log_folder=log_folder,
            model_file=log_folder / f'diffnet_model.pt',
            metadata_file=log_folder / f'metadata.pt',
            **kwargs,
        )

    def restamp(self, idx: int, mkdir=True):
        kwargs = self.__dict__.copy()
        for k in ('opts', 'log_folder', 'model_file', 'metadata_file'):
            kwargs.pop(k)
        return Metadata.stamped(opts=self.opts, idx=idx, mkdir=mkdir, **kwargs)


@dataclass
class TrainedSpec:
    trained_file: Path
    idx: int


@dataclass
class Options:
    mesh_file_mode: str
    data_file: Path
    data_dir: Path
    log_folder: Path
    log_file: Path
    trained: TrainedSpec | None

    input_features: tuple[FeatureMode]
    channel: tuple[int | Sequence[int]]
    k_eig: tuple[int]
    learning_rate: tuple[float]
    decay_every: tuple[int]
    decay_rate: tuple[float]
    dropout: tuple[bool]
    spike_window: tuple[tuple[float, float]]
    weight_error: tuple[WeightErrorMode | None]
    n_faces: tuple[None | int]
    use_visible: tuple[UseVisibleMode | None]
    use_color: tuple[UseColorMode | None]
    norm_verts: tuple[NormVertMode | None]
    augment: tuple[AugmentMode | None]
    n_blocks: tuple[int, ...]
    ultimate_linear: tuple[bool, ...]

    train_frac: float = 0.95
    n_epoch: int = 1
    iter_channels: bool = False

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

    def metadata(self, idx: int, **kwargs) -> Metadata:
        return Metadata.stamped(opts=self, idx=idx, mkdir=False, **kwargs)

    def iter_metadata(self) -> Iterator[Metadata]:
        hparams = {k: v for k, v in self.__dict__.items() if isinstance(v, tuple)}
        for idx, hp in enumerate(hparam_combinations(hparams)):
            meta = self.metadata(idx=idx, **hp)
            if self.iter_channels:
                for ch_idx in range(len(meta.channel)):
                    yield replace(meta, isolate_channel_idx=ch_idx)
            else:
                yield meta

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

        logging.getLogger('matplotlib').setLevel('INFO')


@dataclass
class Experiment:
    metadata: Metadata
    model: DiffusionNet
    device: torch.device
    optimizer: torch.optim.Adam
    train_dataset: GaDataset
    test_dataset: GaDataset
    writer: SummaryWriter

    def load_item(self, data: MeshData):
        verts = data.verts.to(self.device)
        faces = data.faces.to(self.device)
        # _frames = frames.to(self.device)
        mass = data.mass.to(self.device)
        L = data.L.to(self.device)
        evals = data.evals.to(self.device)
        evecs = data.evecs.to(self.device)
        gradX = data.gradX.to(self.device)
        gradY = data.gradY.to(self.device)

        if (weights := data.weight) is not None:
            if not isinstance(weights, torch.Tensor):
                weights = torch.Tensor(np.asarray(weights))
            weights = weights.to(self.device)

        labels = data.labels
        if labels is not None:
            labels = torch.tensor(labels)
            labels = labels.to(self.device)

        # Construct features
        match self.metadata.input_features:
            case 'xyz':
                features = verts
            case 'hks' | ('dirac', _):
                features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)
            case x:
                raise NotImplementedError(x)

        if self.metadata.use_visible:
            features = torch.cat([features, data.visible.reshape(-1, 1).to(self.device)], dim=1)

        if self.metadata.use_color:
            features = torch.cat([features, data.color.to(self.device)], dim=1)

        # Apply the model
        preds = self.model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
        return labels, preds, weights

    def train_epoch(self, loader: DataLoader, epoch: int) -> tuple[float, ScatterData]:
        # Returns mean training loss
        if epoch > 0 and epoch % self.metadata.decay_every == 0:
            self.metadata.curr_learning_rate *= self.metadata.decay_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.metadata.curr_learning_rate

        self.model.train()
        self.optimizer.zero_grad()
        losses = []
        all_obs = []
        all_preds = []

        for data in tqdm(loader, leave=False):
            labels, preds, weights = self.load_item(data)

            err = torch.square(preds - labels)
            if weights is not None:
                err = err * weights

            loss = torch.mean(err)
            loss.backward()
            losses.append(toNP(loss))

            all_obs.append(labels.detach().cpu().numpy())
            all_preds.append(preds.detach().cpu().numpy())

            # Step the optimizer
            self.optimizer.step()
            self.optimizer.zero_grad()

        sd = ScatterData(
            obs=np.stack(all_obs, axis=0),
            preds=np.stack(all_preds, axis=0),
        )
        return float(np.mean(losses)), sd

    def test(self, loader):
        self.model.eval()
        losses = []

        all_obs = []
        all_preds = []

        with torch.no_grad():
            for data in tqdm(loader, leave=False):
                labels, preds, weights = self.load_item(data)
                err = torch.square(preds - labels)
                if weights is not None:
                    err = err * weights

                loss = torch.mean(err)
                losses.append(toNP(loss))

                all_obs.append(labels.detach().cpu().numpy())
                all_preds.append(preds.detach().cpu().numpy())

        sd = ScatterData(
            obs=np.stack(all_obs, axis=0),
            preds=np.stack(all_preds, axis=0),
        )
        return float(np.mean(losses)), sd

    def predict(self, loader: DataLoader, agg_fn: Any = np.concatenate):
        self.model.eval()
        obs, preds = [], []

        with torch.no_grad():
            for data in tqdm(loader):
                _obs, _preds, _weights = self.load_item(data)
                obs.append(_obs.cpu().numpy())
                preds.append(_preds.cpu().numpy())

        return agg_fn(obs), agg_fn(preds)

    def load_mesh_img(
            self,
            dataset: GaDataset,
            stim_idx: int,
            ch_idx: int,
            upsample=False,
            background_color=None,
    ):
        import PIL
        import pyvista as pv

        r = dataset.df.iloc[stim_idx]
        opts = self.metadata.opts
        render_img = PIL.Image.open(opts.data_dir / r.render)  # noqa
        m_full0 = pv.read(opts.data_dir / r.remeshed)
        m_simp0 = pv.read(opts.data_dir / r.simplified)

        tr = np.linalg.inv(m_full0.field_data['to_cam_transform'])
        m_simp1 = m_simp0.transform(tr, inplace=False)
        m_full1 = m_full0.transform(tr, inplace=False)

        _orig = self.model.outputs_at
        self.model.outputs_at = 'vertices'
        _, vert_weights, _ = self.load_item(dataset[stim_idx])
        m_simp1.point_data['x'] = vert_weights[:, ch_idx].cpu().detach().numpy()
        self.model.outputs_at = _orig

        if upsample:
            mesh = m_full1.interpolate(m_simp1)
        else:
            mesh = m_simp1

        p = pv.Plotter(window_size=(1024, 1024))  # noqa
        if background_color:
            p.background_color = background_color
        p.add_mesh(mesh, scalars='x', show_scalar_bar=False)
        p.camera.position = m_full0.field_data['cam_pos']
        p.camera.focal_point = m_full0.field_data['cam_focal_point']
        p.camera.up = m_full0.field_data['cam_view_up']
        mesh_img = PIL.Image.fromarray(p.screenshot())  # noqa

        return p, mesh, render_img, mesh_img


@dataclass
class DataSpec:
    data_file: Path
    channel: tuple[int, ...]
    trained: TrainedSpec | None = None

    def all_channels(self) -> tuple[tuple[int, ...]]:
        return self.channel,

    def split_channels(self, include_all_channels: bool, channel_subset=None) -> tuple[tuple[int], ...]:
        channels = self.channel
        if channel_subset is not None:
            channels = tuple(np.array(channels)[channel_subset])

        split = tuple((c,) for c in channels)
        if include_all_channels:
            return split + (channels,)
        else:
            return split


def specs():
    return {
        9: DataSpec(
            data_file=Path(r"D:\resynth\run_09_10\run00009_resynth\run00009_resynth.hdf"),
            channel=(29, 2, 19, 31, 0, 23, 12, 14, 18, 8),
            # trained=TrainedSpec(Path(r"D:\resynth\run_09_10\run00009_resynth\2025-08-08-12-28-22\opts_and_metadata.pt"), 5),
        ),
        20: DataSpec(
            data_file=Path(r"D:\resynth\run_20_21\run00020_resynth\run00020_resynth.hdf"),
            channel=(2, 17, 13, 29, 14, 7, 23, 3, 28, 8, 12, 18, 31, 27, 4, 11, 30, 19, 20, 24),
            trained=None,
        ),
        38: DataSpec(
            data_file=Path(r"D:\resynth\run_38_39\run00038_resynth\run00038_resynth.hdf"),
            channel=(14, 17, 29, 23, 2, 0, 13, 31, 3, 26, 9, 20, 11, 18),
            trained=None,
        ),
        42: DataSpec(
            data_file=Path(r"D:\resynth\run_42_43\run00042_resynth\run00042_resynth.hdf"),
            channel=(18, 9, 7, 28, 24, 27, 5, 22, 19, 10, 26, 20, 11),
            trained=None,
        ),
        48: DataSpec(
            data_file=Path(r"D:\resynth\run_48_49\run00048_resynth\run00048_resynth.hdf"),
            channel=(14, 17, 29, 23, 2, 0, 13, 31, 3, 26, 28, 9, 20, 11, 18),
            trained=None,
        ),
        51: DataSpec(
            data_file=Path(r"D:\resynth\run_51_52\run00051_resynth\run00051_resynth.hdf"),
            channel=(0, 2, 29, 5, 17, 23, 14, 31, 18, 30, 7, 25, 3, 9),
            trained=None,
        ),
    }



def main():
    logger = logging.getLogger(__name__)

    augment = (
        None,
        AugmentMode(desc='all', max_rotate=np.deg2rad(30), max_translate=0.1, max_scale=0.1),
        AugmentMode(desc='rot', max_rotate=np.deg2rad(30), max_translate=None, max_scale=None),
        AugmentMode(desc='translate', max_rotate=None, max_translate=0.1, max_scale=None),
        AugmentMode(desc='scale', max_rotate=None, max_translate=None, max_scale=0.1),
    )
    spec = specs()[51]

    opts = Options.for_timestamp(
        n_epoch=75, # 75,
        mesh_file_mode='simplified',
        train_frac=0.90,

        data_file=spec.data_file,
        # channel=spec.split_channels(include_all_channels=True, channel_subset=[0, 4, 5, 6, 7, 10]),      # !!!!!
        channel = spec.all_channels(),
        trained=spec.trained,               # !!!!!
        iter_channels=False,                # !!!!!
        ultimate_linear=(False,),

        spike_window=((0.07, 0.75),),  # ) (0.07, 0.4), (0.4, 0.75)),
        weight_error=(None,),
        augment=augment,  # (None, augment)
        k_eig=(128,),
        learning_rate=(1e-4,),
        decay_every=(10,),
        decay_rate=(0.5,),
        input_features=('xyz',),  #
        use_visible=(None,),
        use_color=(None,),
        norm_verts=(None,),
        n_blocks=(4,),  # (3, 4, 5),
        dropout=(False,),
        n_faces=(500,),
    )
    opts.init_log()
    train_test_scenes = None
    orig_metas = list(opts.iter_metadata())
    stamped_metas = []

    if t := spec.trained:
        f = torch.load(t.trained_file)['metadata'][t.idx].model_file
        starting_weights = torch.load(f)
    else:
        starting_weights = None

    for i, meta in enumerate(tqdm(orig_metas)):
        meta = meta.restamp(idx=i)
        stamped_metas.append(meta)

        train_dataset, test_dataset = meta.load_datasets(train_test_scenes=train_test_scenes)
        train_test_scenes = train_dataset.df.index.values, test_dataset.df.index.values
        expt = meta.experiment(train_dataset=train_dataset, test_dataset=test_dataset)
        if starting_weights:
            expt.model.load_state_dict(starting_weights)

        train_loader = DataLoader(expt.train_dataset, batch_size=None, shuffle=True)
        test_loader = DataLoader(expt.test_dataset, batch_size=None)

        best_loss = np.inf

        for epoch in (pbar := tqdm(range(opts.n_epoch))):
            train_loss, train_sd = expt.train_epoch(train_loader, epoch)
            expt.writer.add_scalar(f'loss/train', train_loss, epoch)
            train_ch_loss = torch.tensor(train_sd.by_channel_loss())
            expt.writer.add_tensor(f'loss/train_by_ch', train_ch_loss, epoch)

            test_loss, test_sd = expt.test(test_loader)
            expt.writer.add_scalar(f'loss/test', test_loss, epoch)
            test_ch_loss = torch.tensor(test_sd.by_channel_loss())
            expt.writer.add_tensor(f'loss/test_by_ch', test_ch_loss, epoch)

            pbar.set_postfix(dict(train=train_loss, test=test_loss))

            if test_loss < best_loss:
                best_loss = test_loss
                # logger.debug('Saving best test loss to %s', meta.model_file)
                torch.save(expt.model.state_dict(), meta.model_file)

        last_model_file = meta.model_file.with_suffix('.last' + meta.model_file.suffix)
        torch.save(expt.model.state_dict(), last_model_file)

    metadata = dict(
        opts=opts,
        metadata=stamped_metas,
        train_scenes=train_test_scenes[0],
        test_scenes=train_test_scenes[1],
    )
    final = opts.log_folder / 'opts_and_metadata.pt'
    torch.save(metadata, final)
    logger.debug("Saving all metadata to %s", final)
    print(f'file = Path(r"{final}")')


@dataclass
class ScatterData:
    obs: np.ndarray
    preds: np.ndarray
    metadata: Metadata | None = None
    scenes: pd.DataFrame | None = None  # noqa
    responses: pd.DataFrame | None = None  # noqa

    def loc(
            self,
            scene_ids: np.ndarray | Sequence[int] | None,  # noqa
            channel: int | None = None,
    ):
        obs, preds = self.obs, self.preds

        if scene_ids is not None:
            idx = self.scenes.index.isin(scene_ids)
            obs, preds = obs[idx, :], preds[idx, :]

        if channel is not None:
            channel_idx = self.metadata.channel.index(channel)
            obs, preds = obs[:, channel_idx], preds[:, channel_idx]
        else:
            obs, preds = obs.reshape(-1), preds.reshape(-1)

        return obs, preds

    def for_scenes(self, scene_ids):
        idx = self.scenes.index.isin(scene_ids)
        return self.obs[idx, :], self.preds[idx, :]

    def by_channel_corr_coeffs(self):
        from scipy.stats import pearsonr
        return np.array([
            pearsonr(o, p).statistic
            for (o, p) in zip(self.obs.T, self.preds.T)
        ])

    def by_channel_loss(self, scene_ids=None):
        obs, preds = self.obs, self.preds
        if scene_ids is not None:
            idx = self.scenes.index.isin(scene_ids)
            obs, preds = obs[idx, :], preds[idx, :]

        return ((obs - preds) ** 2).mean(axis=0)


class Reader:
    def __init__(self, metadata: Metadata, train_scenes, test_scenes):
        self.reader = SummaryReader(str(metadata.log_folder))
        self._meta = metadata
        self.train_scenes, self.test_scenes = train_scenes, test_scenes
        self._experiment: Experiment | None = None

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
    def scalar(self, tag: str) -> tuple[np.ndarray, np.ndarray]:
        df = self.scalars
        df = df[df.tag == tag]
        return df.step.values, df.value.values  # noqa

    @cached_property
    def tensors(self):
        return self.reader.tensors

    @lru_cache
    def tensor(self, tag: str) -> tuple[np.ndarray, np.ndarray]:
        df = self.tensors
        df = df[df.tag == tag]
        epoch = df.step.values
        values = np.stack(df.value)
        return epoch, values  # noqa

    def format_hparams(self, tags=None):
        df = self.reader.hparams.set_index('tag')
        if not tags:
            tags = [k for k, v in df.items() if len(set(v)) > 1 and k != 'logfile']

        return ', '.join(f'{k} = {df.loc[k].value}' for k in tags)

    @property
    def metadata(self) -> Metadata:
        return self._meta

    def experiment(self, last_trained: bool = False, outputs_at: str | None = None):
        if self._experiment:
            return self._experiment
        train_ds, test_ds = self._meta.load_datasets(train_test_scenes=(self.train_scenes, self.test_scenes))
        expt = self._experiment = self.metadata.experiment(train_dataset=train_ds, test_dataset=test_ds, )
        f = self._meta.model_file
        if last_trained:
            f = f.with_suffix('.last' + f.suffix)

        expt.model.load_state_dict(torch.load(f))
        if outputs_at:
            expt.model.outputs_at = outputs_at

        return expt

    def load_scatter_data(self, last_trained: bool = False) -> ScatterData:
        m = self._meta

        f = m.log_folder / 'predictions.pt'
        if last_trained:
            f = f.with_suffix('.last' + f.suffix)

        if f.exists():
            d = torch.load(f)
            return ScatterData(
                metadata=self._meta, obs=d['obs'], preds=d['preds'], scenes=d['scenes'], responses=d['responses'])

        dataset = self.metadata.load_dataset(weights=None, augment=None)
        expt = self.experiment(last_trained=last_trained)
        expt.model.outputs_at = 'global_mean'
        loader = DataLoader(dataset, batch_size=None, shuffle=False)
        obs, preds = expt.predict(loader, agg_fn=np.stack)
        d = dict(obs=obs, preds=preds, scenes=dataset.df, responses=dataset.responses)
        torch.save(d, f)
        return ScatterData(metadata=self._meta, obs=obs, preds=preds, scenes=dataset.df, responses=dataset.responses)  # noqa

    @cached_property
    def scatter_data(self) -> ScatterData:
        return self.load_scatter_data(last_trained=False)

    def scatter_plots(self):
        from matplotlib import pyplot as plt

        n_ch = len(self.metadata.channel)
        n = int(np.sqrt(n_ch))
        fig, axs = plt.subplots(n, n, sharex=True, sharey=True, figsize=(n * 5, n * 5))
        axs = axs.reshape(-1)
        sd = self.scatter_data

        for i, ax in enumerate(axs[:n_ch]):
            ax.plot(sd.obs[:, i], sd.preds[:, i], 'k.')
            ax.set_title(f'Ch {i}')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])

        for ax in axs[n_ch:]:
            ax.set_visible(False)

    def scatter_plot(self, channel: int | None = None, axs=None, last_trained: bool = False):
        from matplotlib import pyplot as plt
        from scipy.stats import pearsonr

        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=(11, 5))

        d = self.load_scatter_data(last_trained=last_trained)

        for ax, scenes, ttl in zip(axs, (self.train_scenes, self.test_scenes), ('Train', 'Test')):
            obs, preds = d.loc(scene_ids=scenes, channel=channel)
            ax.plot(obs, preds, 'k.')
            stats = pearsonr(obs, preds)
            ax.set_title(f'{ttl} (r = {stats.statistic:.2f})')
            ax.set_xlabel('Observed response')

        axs[0].set_ylabel('Predicted response')
        return axs

    def plot_training(self, ax=None, mode: Literal['loss', 'corr'] = 'loss'):
        from matplotlib import pyplot as plt

        if ax is None:
            ax = plt.gca()

        ax.plot(*self.scalar(f'{mode}/train'), label='train')
        ax.plot(*self.scalar(f'{mode}/test'), label='test')
        if mode == 'loss':
            ax.set_yscale('log')
        ax.set_ylabel('MSE') if mode == 'loss' else "Pearson's R"
        ax.set_xlabel('Epoch')

    def plot_channel_training(self, figsize=(12, 5), sharey=True, legend=(.92, .35)):
        from matplotlib import pyplot as plt

        ch_idx = self.metadata.isolate_channel_idx
        assert ch_idx is not None

        fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=sharey)  # noqa

        axs[0].set_title('Train')
        hs = axs[0].plot(*self.tensor('loss/train_by_ch'))
        for i, h in enumerate(hs):
            h.set_label(f'Ch {i}')
            h.set_linewidth(2 if i == ch_idx else 0.5)

        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('MSE Loss')


        axs[1].set_title('Test')
        hs = axs[1].plot(*self.tensor('loss/test_by_ch'))
        for i, h in enumerate(hs):
            h.set_linewidth(2 if i == ch_idx else 0.5)

        axs[1].set_xlabel('Epoch')
        plt.suptitle(f'Training channel {ch_idx}')

        if legend not in (False, None):
            if legend is True:
                fig.legend()
            else:
                fig.legend(loc=legend)

        return fig, axs

    def best_test_epoch(self) -> tuple[int, float]:
        _, loss = self.scalar('loss/test')
        i = loss.argmin()
        return i, loss[i]  # noqa


class Readers:
    def __init__(self, readers: list[Reader]):
        self.readers = readers

    def __getitem__(self, i: int | str):
        if isinstance(i, int):
            return self.readers[i]
        elif isinstance(i, str):
            return next(r for r in self.readers if r.log_path == i)
        else:
            readers = list(np.array(self.readers)[i])
            return Readers(readers)

    def __iter__(self):
        yield from self.readers

    @staticmethod
    def load_experiments_df(folder: Path):
        df = cast(pd.DataFrame, pd.read_hdf(Path(folder) / 'experiments.hdf', 'experiments'))
        # df1 = df.query("""
        #         spike_window == '0.03, 0.75'
        #         and weight_mse
        #     """.replace('\n', ' ')
        #                ).sort_values('best_loss_train')
        return df

    def tags(
            self,
            exclude: Sequence[str] = ('log_folder', 'model_file', 'metadata_file', 'curr_learning_rate'),
    ) -> list[str]:
        return [  # type: ignore
            k for k, v in self.hparams.items()
            if (len(set(v)) > 1 and k not in exclude)
        ]

    def labels(self, tags: Sequence[str] | None = None) -> Iterator[str]:
        tags = self.tags() if tags is None else tags

        for (_, hparams) in self.hparams.loc[:, tags].iterrows():
            yield ', '.join(f'{k}={hparams[k]}' for k in tags)

    def scatter_plots(self, tags=None):
        from matplotlib import pyplot as plt
        n = len(self.readers)
        fig, axs = plt.subplots(n, 2, figsize=(10, 5 * n), squeeze=False)

        for i, (r, label, axs_i) in enumerate(zip(self.readers, self.labels(tags=tags), axs)):
            r.scatter_plot(axs=axs_i)
            axs_i[0].set_ylabel(f'{i}) {label})')

        fig.supxlabel('Observed')
        fig.supylabel('Predicted')

        fig.tight_layout()

    def plot_training(
            self,
            tags: Sequence[str] | None = None,
            legend: tuple[float, float] | None = (0.9, 0.05),
            sharey=True,
            figsize=(12, 5),
            mode: Literal['loss', 'corr'] = 'loss',
    ):
        from matplotlib import pyplot as plt

        fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=sharey)
        axs[0].set_title('Train')
        axs[1].set_title('Test')

        for r, label in zip(self.readers, self.labels(tags=tags)):
            axs[0].plot(*r.scalar(f'{mode}/train'), label=label)
            axs[1].plot(*r.scalar(f'{mode}/test'))

        for ax in axs:
            if mode == 'loss':
                ax.set_yscale('log')
            ax.set_ylabel('MSE Loss' if mode == 'loss' else "Pearson's R")
            ax.set_xlabel('Epoch')

        if legend not in (False, None):
            if legend is True:
                fig.legend()
            else:
                fig.legend(loc=legend)

        return fig, axs

    @cached_property
    def hparams(self):
        h = pd.DataFrame([r.metadata.as_series() for r in self.readers])
        for k, v in h.items():
            if v.dtype.name == 'float64' and (v == v.round()).all():
                h[k] = v.astype('int')
        return h

    @staticmethod
    def from_file(f: Path):
        d = torch.load(f)
        metas = d['metadata']
        train_scenes, test_scenes = d['train_scenes'], d['test_scenes']

        readers = [
            Reader(metadata=m, train_scenes=train_scenes, test_scenes=test_scenes)
            for m in metas
        ]
        return Readers(readers=readers)

    def test_train_corrs(self):
        from matplotlib import pyplot as plt
        from scipy.stats import pearsonr

        fig, ax = plt.subplots(layout='constrained')
        labels = list(self.labels(tags=None))
        corrs = dict(test=[], train=[])

        for r in self:
            sd = r.scatter_data
            for mode, scenes in zip(('train', 'test'), (r.train_scenes, r.test_scenes)):
                obs, preds = sd.loc(scene_ids=scenes)
                corrs[mode].append(pearsonr(obs, preds).statistic)

        x = np.arange(len(labels))
        width = 0.25

        for i, (mode, rvals) in enumerate(corrs.items()):
            _rects = ax.barh(x + width * i, rvals, width, label=mode)

        _ = ax.set_xlabel("Pearson's r")
        _ = ax.set_yticks(x + width / 2, labels)
        _ = ax.legend(loc='best')
        return fig, ax


if __name__ == '__main__':
    main()