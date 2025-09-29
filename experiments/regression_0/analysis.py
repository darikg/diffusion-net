from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Sequence, Literal, cast, Iterator

import numpy as np
import pandas as pd
import torch
from pandas import Series
from tbparse import SummaryReader
from torch.utils.data import DataLoader

from ga_regression import Metadata, Experiment


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
    def __init__(
            self,
            metadata: Metadata,
            train_scenes: np.ndarray,
            test_scenes: np.ndarray,
            trained_file: Path | None = None,
            trained_idx: int | None = None,
    ):
        self.reader = SummaryReader(str(metadata.log_folder))
        self._meta = metadata
        self.train_scenes, self.test_scenes = train_scenes, test_scenes
        self._experiment: Experiment | None = None
        self.trained_file = trained_file
        self.trained_idx = trained_idx

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
        # if self._experiment:
        #     return self._experiment
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
            Reader(
                metadata=m,
                train_scenes=train_scenes,
                test_scenes=test_scenes,
                trained_file=f,
                trained_idx=i,
            )
            for i, m in enumerate(metas)
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
