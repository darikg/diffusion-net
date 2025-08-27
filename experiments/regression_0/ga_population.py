from __future__ import annotations

import PIL
import numpy as np
import pyvista as pv
from matplotlib import pyplot as plt
from scipy.stats import linregress
from sklearn.decomposition import PCA
from tqdm.notebook import tqdm

from ga_dataset import *
from ga_regression import *


class SourceModel:
    def __init__(
            self,
            trained_file: str,
            trained_idx: int,
            cp_data_file: str,
    ):
        self.trained_file = trained_file
        self.trained_idx = trained_idx
        self.cp_data_file = cp_data_file
        self._cp_ndata: NeurophysData | None = None
        self.target: np.ndarray | None = None  # noqa
        self._reader = None

    @property
    def reader(self):
        if not self._reader:
            self._reader = self.load_reader()
        return self._reader

    @property
    def n_channels(self):
        return len(self.reader.metadata.channel)

    def __enter__(self):
        _ = self.reader
        return self

    def __exit__(self, *args, **kwargs):
        self._reader = None

    def load_reader(self):
        return Readers.from_file(Path(self.trained_file))[self.trained_idx]

    def ga_ndata(self):
        return NeurophysData.load_data(
            data_file=Path(self.reader.metadata.opts.data_file),
            file_mode=self.reader.metadata.opts.mesh_file_mode,
            spike_window=self.reader.metadata.spike_window,
            n_faces=self.reader.metadata.n_faces,
            features=None,
            n_min_reps=3,
        )

    def load_corpus_data(
            self,
            model_idx: int,
    ):
        ga_ndata = self.ga_ndata()
        ga_responses = ga_ndata.responses.loc[:, self.reader.metadata.channel]
        ga_r_min = ga_responses.min(axis=0)
        ga_r_max = ga_responses.max(axis=0)

        cp_ndata = self._cp_ndata = NeurophysData.load_data(
            data_file=Path(self.cp_data_file),
            file_mode='thumbnail',
            spike_window=self.reader.metadata.spike_window,
            n_faces=None,
            features=None,
            n_min_reps=None,
        )

        # Normalize to ga min/max
        cp_responses = (cp_ndata.responses.loc[:, self.reader.metadata.channel] - ga_r_min) / (ga_r_max - ga_r_min)
        cp_responses.replace([np.inf, -np.inf], 0, inplace=True)

        # Replace scene ids with filenames
        cp_responses = cp_responses.set_index(cp_ndata.scenes.filename.loc[cp_responses.index])

        # Update column indices
        tuples = [(model_idx, i) for i in range(self.n_channels)]
        cp_responses.columns = pd.MultiIndex.from_tuples(tuples, names=["model_idx", "channel_idx"])
        return cp_responses.fillna(0)

    @cached_property
    def channel_tweaks(self):
        obs, preds = self.reader.scatter_data.for_scenes(scene_ids=self.reader.train_scenes)
        return [
            linregress(preds[:, i], obs[:, i])
            for i in range(self.n_channels)
        ]

    def correct_predictions(self, preds: np.ndarray):
        out = np.zeros_like(preds)
        for i, lr in enumerate(self.channel_tweaks):
            out[:, i] = lr.intercept + lr.slope * preds[:, i]
        return out


class SourceModels:
    def __init__(self, models: list[SourceModel]):
        self.models = models

    def __iter__(self):
        yield from self.models

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.models[i]
        else:
            return SourceModels(list(np.array(self.models)[i]))

    def __len__(self):
        return len(self.models)

    @cached_property
    def cp_responses(self):
        cp_responses = pd.concat([
            tm.load_corpus_data(model_idx=i)
            for i, tm in enumerate(self)
        ], axis=1)

        cp_responses = cp_responses.dropna()
        return cp_responses

    @property
    def cp_l2_mag(self):
        return np.sqrt((self.cp_responses ** 2).sum(axis=1))

    def cp_response_pca(self):
        cp_responses = self.cp_responses
        l2_mag = self.cp_l2_mag

        # best_stim = cp_responses.iloc[l2_mag.argmax()]
        pca = PCA(n_components=2)
        pca.fit(cp_responses)
        x = pca.transform(cp_responses)
        plt.scatter(x[:, 0], x[:, 1], 12, l2_mag)

    @property
    def cp_best_stim(self):
        return self.cp_responses.iloc[self.cp_l2_mag.argmax()]

    def diffnet_ga_spec(self):
        for tm, (model_idx, target) in zip(self.models, self.cp_best_stim.groupby('model_idx')):
            tm.target = target.values
            s = f"""
            - (): cemetery.diffnet.TrainedModel
              file: '{tm.trained_file}'
              idx: {tm.trained_idx}
              target: [{', '.join(f'{x:.04f}' for x in target.values)}]"""
            print(s[1:])

    def best_cp_images(self, n: int):
        # Returns filenames...
        return self.cp_l2_mag.sort_values(ascending=False).head(n).index

    def plot_best_cp_images(self, grid_sz=(2, 5)):
        nr, nc = grid_sz
        best_imgs = self.best_cp_images(nr * nc)
        sz = 5
        d = self[0]._cp_ndata  # noqa
        filenames = d.scenes.set_index('filename').thumbnail.loc[best_imgs]
        img_files = [d.data_file.parent / f for f in filenames]
        fig, axs = plt.subplots(nr, nc, figsize=(nc * sz, nr * sz))

        for ttl, img_file, ax in zip(best_imgs, img_files, axs.reshape(-1)):
            img = PIL.Image.open(img_file)  # noqa
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(ttl.split('.')[0])

    def get_predictions(self, probe_meshes: list[ProbeMesh], correct_channels: bool):
        n_mesh = len(probe_meshes)
        out = []
        for i, tm in enumerate(tqdm(self.models)):
            with tm:
                expt = tm.reader.experiment()
                n_channel = len(expt.metadata.channel)
                preds = np.full((n_mesh, n_channel), np.nan)
                expt.model.eval()

                with torch.no_grad():
                    for j, pm in enumerate(probe_meshes):
                        md = pm.mesh_data()
                        _, preds_j, _ = expt.load_item(md)
                        preds[j, :] = preds_j.cpu().numpy()

                if correct_channels:
                    preds = tm.correct_predictions(preds)

            tuples = [(i, j) for j in range(n_channel)]
            columns = pd.MultiIndex.from_tuples(tuples, names=["model_idx", "channel_idx"])
            out.append(pd.DataFrame(preds, columns=columns))

        return pd.concat(out, axis=1)

    @staticmethod
    def current_trained():
        models = [
            SourceModel(
                trained_file=r"D:\resynth\run_09_10\run00009_resynth\2025-08-08-12-28-22\opts_and_metadata.pt",
                trained_idx=5,
                cp_data_file=r"D:\resynth\run_09_10\run00010_exported\run00010_exported.hdf",
            ),
            SourceModel(
                trained_file=r"D:\resynth\run_20_21\run00020_resynth\2025-08-09-07-18-37\opts_and_metadata.pt",
                trained_idx=5,
                cp_data_file=r"D:\resynth\run_20_21\run00021_exported\run00021_exported.hdf",
            ),
            SourceModel(
                trained_file=r"D:\resynth\run_38_39\run00038_resynth\2025-08-09-17-21-56\opts_and_metadata.pt",
                trained_idx=5,
                cp_data_file=r"D:\resynth\run_38_39\run00039_exported\run00039_exported.hdf",
            ),
            SourceModel(
                trained_file=r"D:\resynth\run_42_43\run00042_resynth\2025-08-11-14-04-05\opts_and_metadata.pt",
                trained_idx=5,
                cp_data_file=r"D:\resynth\run_42_43\run00043_exported\run00043_exported.hdf",
            ),
            SourceModel(
                trained_file=r"D:\resynth\run_48_49\run00048_resynth\2025-08-11-06-09-16\opts_and_metadata.pt",
                trained_idx=5,
                cp_data_file=r"D:\resynth\run_48_49\run00049_exported\run00049_exported.hdf",
            ),
            SourceModel(
                trained_file=r"D:\resynth\run_51_52\run00051_resynth\2025-08-07-12-07-53\opts_and_metadata.pt",
                trained_idx=5,
                cp_data_file=r"D:\resynth\run_51_52\run00052_exported\run00052_exported.hdf",
            ),
        ]
        return SourceModels(models)

@contextmanager
def maybe_plotter(p=None, **kwargs) -> Iterator:
    import pyvista as pv
    show = p is None
    p = p or pv.Plotter(**kwargs)
    yield p
    if show:
        p.show()

def iter_subplots(
        plotter=None,
        link_views=True,
        **kwargs
) -> Iterator[Tuple[Tuple[int, int], Plotter]]:
    with maybe_plotter(plotter, **kwargs) as p:
        nr, nc = p.shape
        for i in range(nr):
            for j in range(nc):
                p.subplot(i, j)
                yield (i, j), p

        if link_views:
            p.link_views()


class ProbeMesh:
    def __init__(
            self,
            mesh: pv.PolyData,
            cam_pos: np.ndarray,
            tgt_pos: np.ndarray,
            corpus_index: str | None = None,
    ):
        self.mesh = mesh
        self.cam_pos = cam_pos
        self.tgt_pos = tgt_pos
        self.corpus_index = corpus_index

    @property
    def camera(self) -> pv.Camera:
        cam = pv.Camera()
        cam.position = self.cam_pos
        cam.focal_point = self.tgt_pos
        cam.SetViewUp(0, 0, 1)
        return cam

    def mesh_to_camera(self):
        t = pv.array_from_vtkmatrix(self.camera.GetViewTransformMatrix())
        return self.mesh.transform(t, inplace=False)

    def plotter(self, ground=True):
        p = pv.Plotter()
        p.add_mesh(self.mesh)
        p.camera = self.camera
        if ground:
            p.add_mesh(pv.Plane(), color='gray')
        return p

    def render(self, ground=True):
        img = self.plotter(ground=ground).screenshot()
        return PIL.Image.fromarray(img)  # noqa

    def scale_camera_dist(self, frac=1.0):
        if frac == 1:
            return self

        cam_vec = self.tgt_pos - self.cam_pos
        # dist = np.linalg.norm(cam_vec)

        mesh = self.mesh.scale(frac, point=self.tgt_pos)
        cam_pos = self.tgt_pos - frac * cam_vec
        return ProbeMesh(mesh=mesh, cam_pos=cam_pos, tgt_pos=self.tgt_pos)

    def mesh_data(self, k_eig: int = 128, op_cache_dir=None):
        mesh = self.mesh_to_camera()
        return MeshData.simple(verts=mesh.points, faces=mesh.regular_faces, k_eig=k_eig, op_cache_dir=op_cache_dir)

    @staticmethod
    def load(
            mesh_file: str | Path,
            cam_pos: tuple[float, float, float],
            tgt_pos: tuple[float, float, float],
            repair=True,
            fill_holes=False,
            cache=True,
            recache=False,
            corpus_index: str | None = None,
    ):
        import pymeshfix  # noqa
        from seagullmesh import Mesh3

        mesh_file = Path(mesh_file)
        cache_file = mesh_file.with_suffix(mesh_file.suffix + '.cache')

        if cache and not recache and cache_file.exists():
            probe_mesh = pv.read(cache_file)
        else:
            pv_mesh = pv.read(mesh_file).triangulate().clean()

            if repair:
                meshfix = pymeshfix.MeshFix(pv_mesh.points, pv_mesh.regular_faces)
                meshfix.repair()
                pv_mesh = meshfix.mesh

            sm_mesh = Mesh3.from_pyvista(pv_mesh)

            if fill_holes:
                sm_mesh.triangulate_holes(sm_mesh.extract_boundary_cycles())

            sm_wrapped = sm_mesh.alpha_wrapping(relative_alpha=100, relative_offset=600)
            probe_mesh = sm_wrapped.to_pyvista()

            if cache:
                probe_mesh.save(cache_file)

        return ProbeMesh(
            mesh=probe_mesh, cam_pos=np.array(cam_pos), tgt_pos=np.array(tgt_pos), corpus_index=corpus_index)



