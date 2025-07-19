from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property, lru_cache
from itertools import product
from pathlib import Path
from typing import Tuple, Any, Sequence, Iterator, cast

import numpy as np
import pandas as pd
import torch
from numpy.random import permutation
from pandas import Series
from tbparse import SummaryReader
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net  # noqa
from diffusion_net.utils import toNP   # noqa
from diffusion_net.layers import DiffusionNet   # noqa
from ga_dataset import GaDataset, UseVisibleMode, WeightErrorMode, MeshData, UseColorMode, NormVertMode  # noqa


def hparam_combinations(hparams: dict[str, Sequence[Any]]) -> Iterator[dict[str, Any]]:
    for vals in product(*hparams.values()):
        yield {k: v for k, v in zip(hparams.keys(), vals)}
        
        
@dataclass
class Metadata:
    opts: Options
    log_folder: Path
    model_file: Path
    metadata_file: Path

    input_features: str
    channel: int | Sequence[int]
    k_eig: int = 128
    learning_rate: float = 1e-3
    decay_every: int = 50
    decay_rate: float = 0.5
    dropout: bool = False
    n_faces: int | None = None
    spike_window: tuple[float, float] = (0.07, 0.3)
    weight_error: WeightErrorMode | None = None
    use_visible: UseVisibleMode | None = None
    use_color: UseColorMode | None = None
    norm_verts: NormVertMode | None = None

    curr_learning_rate: float | None = None

    def __post_init__(self):
        self.curr_learning_rate = self.learning_rate

    def load_data(self):
        scenes, responses, op_cache_dir, weights, fit_fns = GaDataset.load_data(
            data_file=self.opts.data_file,
            channel=self.channel,
            file_mode=self.opts.mesh_file_mode,
            spike_window=self.spike_window,
            weight_error=self.weight_error,
            n_faces=self.n_faces,
        )
        return scenes, responses, op_cache_dir, weights, fit_fns

    def load_datasets(
            self,
            train_test_scenes: tuple[Sequence[int], Sequence[int]] | None = None,
    ) -> Tuple[GaDataset, GaDataset]:
        scenes, responses, op_cache_dir, weights, fit_fns = self.load_data()

        if train_test_scenes:
            train_scenes, test_scenes = train_test_scenes
            train_idxs = np.isin(scenes.scene, train_scenes)
            test_idxs = np.isin(scenes.scene, test_scenes)
        else:
            n = len(scenes)
            idx = permutation(n)
            split = int(n * self.opts.train_frac)
            train_idxs = idx < split
            test_idxs = idx >= split

        train_dataset, test_dataset = (
            GaDataset(
                df=scenes[idx],
                responses=responses[idx],
                root_dir=self.opts.data_file.parent,
                k_eig=self.k_eig,
                op_cache_dir=op_cache_dir,
                file_mode=self.opts.mesh_file_mode,
                weights=weights[idx] if weights is not None else None,
                use_visible=self.use_visible,
                use_color=self.use_color,
                norm_verts=self.norm_verts,
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

        return DiffusionNet(
            C_in=C_in,
            C_out=n_channels_out,
            C_width=64,
            N_block=4,
            last_activation=None,
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


@dataclass
class Options:
    mesh_file_mode: str
    data_file: Path
    data_dir: Path
    log_folder: Path
    log_file: Path

    input_features: tuple[str]
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

    train_frac: float = 0.95
    n_epoch: int = 1

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

    def metadata(self, **kwargs) -> Metadata:
        stamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        log_folder = self.log_folder / stamp
        log_folder.mkdir(parents=True, exist_ok=True)
        return Metadata(
            opts=self,
            log_folder=log_folder,
            model_file=log_folder / f'diffnet_model_{stamp}.pt',
            metadata_file=log_folder / f'metadata_{stamp}.pt',
            **kwargs,
        )

    def iter_metadata(self) -> Iterator[Metadata]:
        hparams = {k: v for k, v in self.__dict__.items() if isinstance(v, tuple)}
        for hp in hparam_combinations(hparams):
            yield self.metadata(**hp)

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

        if not isinstance((labels := data.labels), torch.Tensor):
            labels = torch.Tensor(labels)

        labels = labels.to(self.device)

        # if augment_random_rotate:
        #     verts = diffusion_net.utils.random_rotate_points(verts)

        # Construct features
        if self.metadata.input_features == 'xyz':
            features = verts
        elif self.metadata.input_features == 'hks':
            features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)
        else:
            raise NotImplementedError(self.metadata.input_features)

        if self.metadata.use_visible:
            features = torch.cat([features, data.visible.reshape(-1, 1).to(self.device)], dim=1)

        if self.metadata.use_color:
            features = torch.cat([features, data.color.to(self.device)], dim=1)

        # Apply the model
        preds = self.model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
        return labels, preds, weights

    def train_epoch(self, loader: DataLoader, epoch: int) -> float:
        # Returns mean training loss
        if epoch > 0 and epoch % self.metadata.decay_every == 0:
            self.metadata.curr_learning_rate *= self.metadata.decay_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.metadata.curr_learning_rate

        self.model.train()
        self.optimizer.zero_grad()
        losses = []

        for data in tqdm(loader):
            labels, preds, weights = self.load_item(data)
            err = torch.square(preds - labels)
            if weights is not None:
                err = err * weights

            loss = torch.mean(err)
            losses.append(toNP(loss))
            loss.backward()

            # Step the optimizer
            self.optimizer.step()
            self.optimizer.zero_grad()

        return np.mean(losses)

    def test(self, loader):
        self.model.eval()
        losses = []

        with torch.no_grad():
            for data in tqdm(loader):
                labels, preds, weights = self.load_item(data)
                err = torch.square(preds - labels)
                if weights is not None:
                    err = err * weights

                loss = torch.mean(err)
                losses.append(toNP(loss))

        return np.mean(losses)

    def predict(self, loader: DataLoader, agg_fn: Any = np.concatenate):
        self.model.eval()
        obs, preds = [], []

        with torch.no_grad():
            for data in tqdm(loader):
                _obs, _preds, _weights = self.load_item(data)
                obs.append(_obs.cpu().numpy())
                preds.append(_preds.cpu().numpy())

        return agg_fn(obs), agg_fn(preds)


def main():
    logger = logging.getLogger(__name__)
    opts = Options.for_timestamp(
        # data_file=Path(r"D:\resynth\run_51_52\1k_faces\run00051_resynth.hdf"),
        # channel=(0, 2, 29, 5, 17, 23, 14, 31, 18, 30, 7, 25, 3, 9),
        # data_file=Path(r"D:\resynth\run_48_49\1k_faces\run00048_resynth.hdf"),
        # channel=(14, 17, 29, 23, 2, 0, 13, 31, 3, 26, 28, 9, 20, 11, 18),
        # data_file=Path(r"D:\resynth\run_42_43\1k_faces\run00042_resynth.hdf"),
        # channel=(14, 17, 29, 23, 2, 0, 13, 31, 3, 26, 28, 9, 20, 11, 18),
        # data_file=Path(r"D:\resynth\run_38_39\1k_faces\run00038_resynth.hdf"),
        # channel=(14, 17, 29, 23, 2, 0, 13, 31, 3, 26, 9, 20, 11, 18),
        # data_file=Path(r"D:\resynth\run_20_21\1k_faces\run00020_resynth.hdf"),
        # channel=(2, 17, 13, 29, 14, 7, 23), # , 3, 28, 8, 12, 18, 31, 27, 4),
        # data_file=Path(r"D:\resynth\run_09_10\1k_faces\run00009_resynth.hdf"),
        # channel=(29, 2, 19, 31, 0, 23, 12, 14, 18, 8),

        # data_file=Path(r"D:\resynth\run_48_49\resynth_everything3\run00048_resynth.hdf"),
        # data_file=Path(r"D:\resynth\run_48_49\many_faces\run00048_resynth.hdf"),
        data_file=Path(r"D:\resynth\run_48_49\run00048_simp_vis_color\run00048_resynth.hdf"),
        n_epoch=250,
        mesh_file_mode='simplified',
        train_frac=0.95,

        channel=((14, 17, 29, 23, 2, 0, 13, 31, 3, 26, 28, 9, 20, 11, 18),),
        input_features=('hks',),
        spike_window=((0.07, 0.75),),  # ) (0.07, 0.4), (0.4, 0.75)),
        weight_error=(None,),
        k_eig=(128,),
        learning_rate=(1e-3,),
        decay_every=(50,),
        decay_rate=(0.5,),
        use_visible=(None,),  # 'orig', 'shuffled'),
        use_color=(None,),
        norm_verts=(None, ('mean', 'max_rad'), ('bbox', 'area')),
        dropout=(False,),
        n_faces=(500,),
    )
    opts.init_log()
    train_test_scenes = None
    metas = []

    for meta in opts.iter_metadata():
        metas.append(meta)

        train_dataset, test_dataset = meta.load_datasets(train_test_scenes=train_test_scenes)
        train_test_scenes = train_dataset.df.scene.values, test_dataset.df.scene.values
        expt = meta.experiment(train_dataset=train_dataset, test_dataset=test_dataset)

        train_loader = DataLoader(expt.train_dataset, batch_size=None, shuffle=True)
        test_loader = DataLoader(expt.test_dataset, batch_size=None)

        best_loss = np.inf

        for epoch in range(opts.n_epoch):
            train_loss = expt.train_epoch(train_loader, epoch)
            expt.writer.add_scalar(f'loss/train', train_loss, epoch)
            test_loss = expt.test(test_loader)
            expt.writer.add_scalar(f'loss/test', test_loss, epoch)
            logger.debug(f"Epoch {epoch}: Train: {train_loss:.5e}  Test: {test_loss:.5e}")

            if test_loss < best_loss:
                best_loss = test_loss
                logger.debug('Saving best test loss to %s', meta.model_file)
                torch.save(expt.model.state_dict(), meta.model_file)

        mf = meta.model_file.with_suffix('.last' + meta.model_file.suffix)
        logger.debug("Saving last model to %s", mf)
        torch.save(expt.model.state_dict(), meta.model_file)

    metadata = dict(
        opts=opts,
        metadata=metas,
        train_scenes=train_test_scenes[0],
        test_scenes=train_test_scenes[1],
    )
    final = opts.log_folder / 'opts_and_metadata.pt'
    torch.save(metadata, final)
    logger.debug("Saving all metadata to %s", final)


if __name__ == '__main__':
    main()


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
    def scalar(self, tag):
        df = self.scalars
        df = df[df.tag == tag]
        return df.step, df.value

    def format_hparams(self, tags=None):
        df = self.reader.hparams.set_index('tag')
        if not tags:
            tags = [k for k, v in df.items() if len(set(v)) > 1 and k != 'logfile']

        return ', '.join(f'{k} = {df.loc[k].value}' for k in tags)

    @property
    def metadata(self) -> Metadata:
        return self._meta

    def experiment(self):
        if self._experiment:
            return self._experiment
        train_ds, test_ds = self._meta.load_datasets(train_test_scenes=(self.train_scenes, self.test_scenes))
        expt = self._experiment = self.metadata.experiment(train_dataset=train_ds, test_dataset=test_ds, )
        expt.model.load_state_dict(torch.load(self._meta.model_file))
        return expt

    @cached_property
    def predictions(self):
        m = self._meta

        f = m.log_folder / 'predictions.pt'
        if f.exists():
            return torch.load(f)

        scenes, responses, op_cache_dir, weights, fit_fns = m.load_data()
        dataset = GaDataset(
            df=scenes,
            responses=responses,
            root_dir=m.opts.data_file.parent,
            k_eig=m.k_eig,
            op_cache_dir=op_cache_dir,
            file_mode=m.opts.mesh_file_mode,
            weights=None,
            use_visible=m.use_visible,
            use_color=m.use_color,
            norm_verts=m.norm_verts,
        )

        expt = self.experiment()
        expt.model.outputs_at = 'global_mean'
        loader = DataLoader(dataset, batch_size=None, shuffle=False)
        obs, preds = expt.predict(loader, agg_fn=np.stack)
        d = dict(obs=obs, preds=preds, scenes=scenes, responses=responses)
        torch.save(d, f)
        return d

    def scatter_plot(self):
        from matplotlib import pyplot as plt
        from scipy.stats import pearsonr

        fig, axs = plt.subplots(1, 2, figsize=(11, 5))
        d_preds = self.predictions

        for ax, scenes, ttl in zip(axs, (self.train_scenes, self.test_scenes), ('Train', 'Test')):
            idx = d_preds['scenes'].scene.isin(scenes)
            obs = d_preds['obs'][idx, :].reshape(-1)
            preds = d_preds['preds'][idx, :].reshape(-1)

            ax.plot(obs, preds, 'k.')
            stats = pearsonr(obs, preds)
            ax.set_title(f'{ttl} (r = {stats.statistic:.2f})')
            ax.set_xlabel('Observed response')

        axs[0].set_ylabel('Predicted response')
        return fig, axs


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

    def labels(self, tags: Sequence[str] | None = None) -> Iterator[str]:
        if tags is None:
            exclude = ('log_folder', 'model_file', 'metadata_file')
            tags = [
                k for k, v in self.hparams.items()
                if (len(set(v)) > 1 and k not in exclude)
            ]

        for (_, hparams) in self.hparams.loc[:, tags].iterrows():
            yield ', '.join(f'{k}={hparams[k]}' for k in tags)

    def scatter_plots(self, tags=None):
        for r, label in zip(self.readers, self.labels(tags=tags)):
            fig, axs = r.scatter_plot()
            fig.suptitle(label)

    def plot_loss(
            self,
            tags: Sequence[str] | None = None,
            legend: tuple[float, float] | None = (0.9, 0.05),
            sharey=True,
            figsize=(12, 5),
    ):
        from matplotlib import pyplot as plt

        fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=sharey)
        axs[0].set_title('Train')
        axs[1].set_title('Test')

        for r, label in zip(self.readers, self.labels(tags=tags)):
            axs[0].plot(*r.scalar('loss/train'), label=label)
            axs[1].plot(*r.scalar('loss/test'))

        for ax in axs:
            ax.set_yscale('log')
            ax.set_ylabel('MSE Loss')
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
