from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Tuple, Any, Sequence, Iterator, TypedDict, TYPE_CHECKING

import click
import numpy as np
import pandas as pd
import torch
from numpy.random import permutation
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm


sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net  # noqa
from diffusion_net.utils import toNP   # noqa
from diffusion_net.layers import DiffusionNet   # noqa
from ga_dataset import GaDataset, UseVisibleMode, WeightErrorMode, MeshData, UseColorMode, NormVertMode, \
    FeatureMode, AugmentMode  # noqa


if TYPE_CHECKING:
    from analysis import ScatterData

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
    num_workers = 2
    persistent_workers = True

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

        kwargs = dict()

        if self.metadata.use_visible:
            visible = data.visible.reshape(-1, 1).to(self.device)
            if self.metadata.use_visible.feature:
                features = torch.cat([features, visible], dim=1)
            if self.metadata.use_visible.multiply:
                kwargs['vertex_weights'] = visible

        if self.metadata.use_color:
            features = torch.cat([features, data.color.to(self.device)], dim=1)

        # Apply the model
        preds = self.model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces, **kwargs)
        return labels, preds, weights

    def train_epoch(self, loader: DataLoader, epoch: int) -> tuple[float, ScatterData]:
        from analysis import ScatterData

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
        from analysis import ScatterData
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

    def predict(self, loader: DataLoader, agg_fn: Any = np.concatenate, leave_pbar=False):
        self.model.eval()
        obs, preds = [], []

        with torch.no_grad():
            for data in tqdm(loader, leave=leave_pbar):
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


def specs(root=r"D:\resynth"):
    root = Path(root)

    return {
        9: DataSpec(
            data_file=root / r"run_09_10/run00009_resynth/run00009_resynth.hdf",
            channel=(29, 2, 19, 31, 0, 23, 12, 14, 18, 8),
            # trained=TrainedSpec(Path(r"D:/resynth/run_09_10/run00009_resynth/2025-08-08-12-28-22/opts_and_metadata.pt"), 5),
        ),
        20: DataSpec(
            data_file=root / r"run_20_21/run00020_resynth/run00020_resynth.hdf",
            channel=(2, 17, 13, 29, 14, 7, 23, 3, 28, 8, 12, 18, 31, 27, 4, 11, 30, 19, 20, 24),
            trained=None,
        ),
        38: DataSpec(
            data_file=root / r"run_38_39/run00038_resynth/run00038_resynth.hdf",
            channel=(14, 17, 29, 23, 2, 0, 13, 31, 3, 26, 9, 20, 11, 18),
            trained=None,
        ),
        42: DataSpec(
            data_file=root / r"run_42_43/run00042_resynth/run00042_resynth.hdf",
            channel=(18, 9, 7, 28, 24, 27, 5, 22, 19, 10, 26, 20, 11),
            trained=None,
        ),
        48: DataSpec(
            data_file=root / r"run_48_49/run00048_resynth/run00048_resynth.hdf",
            channel=(14, 17, 29, 23, 2, 0, 13, 31, 3, 26, 28, 9, 20, 11, 18),
            trained=None,
        ),
        51: DataSpec(
            data_file=root / r"run_51_52/run00051_resynth/run00051_resynth.hdf",
            channel=(0, 2, 29, 5, 17, 23, 14, 31, 18, 30, 7, 25, 3, 9),
            trained=None,
        ),
    }



def run_one_metadata_expt(
        meta: Metadata,
        train_test_scenes: tuple[np.ndarray, np.ndarray],
        starting_weights: torch.Tensor | None,  # noqa
):
    train_dataset, test_dataset = meta.load_datasets(train_test_scenes=train_test_scenes)
    expt = meta.experiment(train_dataset=train_dataset, test_dataset=test_dataset)

    if starting_weights is None and (t := meta.opts.trained):
        f = torch.load(t.trained_file)['metadata'][t.idx].model_file
        starting_weights = torch.load(f)

    if starting_weights is not None:
        expt.model.load_state_dict(starting_weights)

    num_workers = meta.opts.num_workers
    persistent_workers = meta.opts.persistent_workers

    train_loader = DataLoader(expt.train_dataset, batch_size=None, shuffle=True, num_workers=num_workers,
                              persistent_workers=persistent_workers)
    test_loader = DataLoader(expt.test_dataset, batch_size=None, shuffle=False, num_workers=num_workers,
                             persistent_workers=persistent_workers)
    best_loss = np.inf

    for epoch in (pbar := tqdm(range(meta.opts.n_epoch))):
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


@click.group()
def cli():
    pass


class GeneratedExpts(TypedDict):
    opts: Options
    metadata: list[Metadata]
    train_scenes: np.ndarray
    test_scenes: np.ndarray


@cli.command
def generate():
    augment = (
        # AugmentMode(desc='mild', max_rotate=np.deg2rad(30), max_translate=0.10, max_scale=0.15),
        # AugmentMode(desc='med', max_rotate=np.deg2rad(45), max_translate=0.15, max_scale=0.2),
        AugmentMode(desc='hot', max_rotate=np.deg2rad(60), max_translate=0.2, max_scale=0.25),
        None,
    )
    use_visible = (
        None,
        UseVisibleMode(shuffled=False, multiply=False, feature=True),
        UseVisibleMode(shuffled=True, multiply=False, feature=True),

        UseVisibleMode(shuffled=False, multiply=True, feature=False),
        UseVisibleMode(shuffled=True, multiply=True, feature=False),

        # UseVisibleMode(shuffled=True, multiply=False),
        # UseVisibleMode(shuffled=False, multiply=True),
        # UseVisibleMode(shuffled=True, multiply=True),
    )

    root = r"/home/darik/resynth"
    # root = r"D:\resynth"
    spec = specs(root=root)[9]

    opts = Options.for_timestamp(
        n_epoch=2,
        mesh_file_mode='simplified',
        train_frac=0.90,

        data_file=spec.data_file,
        # channel=spec.split_channels(include_all_channels=True, channel_subset=[0, 4, 5, 6, 7, 10]),      # !!!!!
        channel = spec.all_channels(),
        trained=spec.trained,
        iter_channels=False,
        ultimate_linear=(False,),

        spike_window=((0.07, 0.75),),  # ) (0.07, 0.4), (0.4, 0.75)),
        weight_error=(None,),
        augment=augment,  # (None, augment)
        k_eig=(128,),
        learning_rate=(1e-4,),
        decay_every=(25,),
        decay_rate=(0.5,),
        input_features=('xyz',),  #
        use_visible=use_visible,
        use_color=(None,),
        norm_verts=(None,),
        n_blocks=(4,),  # (3, 4, 5),
        dropout=(False,),
        n_faces=(500,),
    )
    opts.init_log()
    train_test_scenes = None
    metas = [meta.restamp(idx=i) for i, meta in  enumerate(opts.iter_metadata())]
    train_dataset, test_dataset = metas[0].load_datasets(train_test_scenes=train_test_scenes)
    train_test_scenes = train_dataset.df.index.values, test_dataset.df.index.values

    metadata = dict(
        opts=opts,
        metadata=metas,
        train_scenes=train_test_scenes[0],
        test_scenes=train_test_scenes[1],
    )
    final = opts.log_folder / 'opts_and_metadata.pt'
    torch.save(metadata, final)

    print(f'{len(metas)} experiments to run')
    print(final)


@cli.command()
@click.argument('file', type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True))
@click.argument('idx', type=int)
def run(
        file: Path,
        idx: int,
):
    g: GeneratedExpts = torch.load(file, weights_only=False)
    run_one_metadata_expt(
        meta=g['metadata'][idx],
        train_test_scenes=(g['train_scenes'], g['test_scenes']),
        starting_weights=None,
    )


@cli.command()
@click.argument('file', type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True))
def submit(file: Path):
    import htcondor  # noqa
    g: GeneratedExpts = torch.load(file, weights_only=False)
    n = len(g['metadata'])
    f = g['opts'].log_folder

    arguments = [
        'run',  '-p', '/home/darik/micromamba/envs/diffnet2', 'python', __file__, 'run',
        str(file),
        '$(ProcId)'
    ]

    job = htcondor.Submit(dict(  # noqa
        executable="/home/darik/micromamba/envs/cemetery/bin/conda",
        arguments = ' '.join(arguments),
        output=str(f / 'condor.$(ProcId).out'),
        error=str(f / 'condor.$(ProcId).err'),
        log=str(f / 'condor.$(ProcId).log'),
        request_cpus="4",
        request_memory="4GB",  # how much memory we want
        Request_GPUs='1',
        # request_disk="128MB",  # how much disk space we want
    ))

    schedd = htcondor.Schedd()  # noqa
    submit_result = schedd.submit(job, count=n)
    print(f'cluster_id = {submit_result.cluster()}')


if __name__ == '__main__':
    cli()