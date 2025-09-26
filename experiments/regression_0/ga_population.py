from __future__ import annotations

import PIL
import numpy as np
import pyvista as pv
from matplotlib import pyplot as plt
from scipy.stats import linregress
from sklearn.decomposition import PCA
from tqdm.autonotebook import tqdm

from ga_dataset import *
from ga_regression import *


class SourceModel:
    def __init__(
            self,
            trained_file: str | Path,
            trained_idx: int,
            cp_data_file: str | Path,
            reader: Reader | None = None,
    ):
        self.trained_file = trained_file
        self.trained_idx = trained_idx
        self.cp_data_file = cp_data_file
        self._cp_ndata: NeurophysData | None = None
        self.target: np.ndarray | None = None  # noqa
        self._reader = reader

    @property
    def reader(self):
        if not self._reader:
            self._reader = self.load_reader()
        return self._reader

    @staticmethod
    def from_reader(r: Reader, cp_data_file: Path):
        return SourceModel(
            trained_file=r.trained_file,
            trained_idx=r.trained_idx,
            cp_data_file=cp_data_file,
            reader=r,
        )

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


class SourceModels:
    def __init__(self, models: list[SourceModel]):
        self.models = models

    def __iter__(self):
        yield from self.models

    def __getitem__(self, i):
        if isinstance(i, (int, np.integer)):
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

    def get_predictions(self, probe_meshes: list[ProbeMesh]):
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

            tuples = [(i, j) for j in range(n_channel)]
            columns = pd.MultiIndex.from_tuples(tuples, names=["model_idx", "channel_idx"])
            out.append(pd.DataFrame(preds, columns=columns))

        df = pd.concat(out, axis=1)
        df = df.set_index(pd.Index([pm.index for pm in probe_meshes]))
        return df

    @staticmethod
    def cp_data_files(run_id: int) -> str:
        return {
             9: r"D:\resynth\run_09_10\run00010_exported\run00010_exported.hdf",
            20: r"D:\resynth\run_20_21\run00021_exported\run00021_exported.hdf",
            38: r"D:\resynth\run_38_39\run00039_exported\run00039_exported.hdf",
            42: r"D:\resynth\run_42_43\run00043_exported\run00043_exported.hdf",
            48: r"D:\resynth\run_48_49\run00049_exported\run00049_exported.hdf",
            51: r"D:\resynth\run_51_52\run00052_exported\run00052_exported.hdf",
        }[run_id]

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

    def get_weights(self, probe_meshes: list[ProbeMesh], expt_kwargs=None) -> list[VertexWeights]:
        expt_kwargs = expt_kwargs or dict()

        n_model, n_probe = len(self), len(probe_meshes)
        weights = np.empty((n_probe, n_model), dtype='O')

        for model_idx, tm in enumerate(tqdm(self.models)):
            expt = tm.reader.experiment(**expt_kwargs)
            expt.model.eval()
            with torch.no_grad():
                for probe_idx, pm in enumerate(probe_meshes):
                    _, preds, _ = expt.load_item(pm.mesh_data())
                    weights[probe_idx, model_idx] = preds.cpu().numpy()

        return [
            VertexWeights(np.concatenate(weights[i, :], axis=1))
            for i in range(len(probe_meshes))
        ]

    def corpus_probe_mesh_preds(self, pms: list[ProbeMesh], cache_file: str | Path, recache=False):
        cache_file = Path(cache_file)
        if recache:
            cache_file.unlink(missing_ok=True)

        if cache_file.exists():
            return pd.read_hdf(cache_file, key=cache_file.stem)
        else:
            cp_pm_preds = self.get_predictions(pms)
            cp_pm_preds.to_hdf(cache_file, key=cache_file.stem)
            return cp_pm_preds

    def corpus_observed_responses(self, cp_pm_preds: pd.DataFrame, cache_file: str | Path, recache=False):
        cache_file = Path(cache_file)

        if recache:
            cache_file.unlink(missing_ok=True)

        if cache_file.exists():
            return pd.read_hdf(cache_file, key=cache_file.stem)
        else:
            cp_responses = self.cp_responses.loc[cp_pm_preds.index]
            cp_responses.to_hdf(cache_file, key=cache_file.stem)
            return cp_responses

@contextmanager
def maybe_plotter(p: pv.Plotter | None = None, **kwargs) -> Iterator[pv.Plotter]:
    show = p is None
    p = p or pv.Plotter(**kwargs)
    yield p
    if show:
        p.show()

def iter_subplots(
        plotter: pv.Plotter | None = None,
        link_views=True,
        **kwargs
) -> Iterator[Tuple[Tuple[int, int], pv.Plotter]]:
    with maybe_plotter(plotter, **kwargs) as p:
        nr, nc = p.shape
        for i in range(nr):
            for j in range(nc):
                p.subplot(i, j)
                yield (i, j), p

        if link_views:
            p.link_views()


def calc_visible(mesh: pv.PolyData, cam: pv.Camera, res: tuple[int, int], off_screen=True) -> np.ndarray:
    from pyvista.plotting import Plotter
    from vtkmodules.vtkRenderingCore import vtkSelectVisiblePoints
    from pyvista.core.filters import _get_output  # noqa

    mesh.point_data['orig_idx'] = np.arange(mesh.n_points)
    p = Plotter(off_screen=off_screen, window_size=res)
    p.add_mesh(mesh)
    p.camera = cam
    if off_screen:
        _img = p.screenshot(window_size=res)
    else:
        p.show()

    svp = vtkSelectVisiblePoints()
    svp.SetInputData(mesh)
    svp.SetRenderer(p.renderer)
    svp.Update()

    vis_mesh = _get_output(svp)
    is_vis = np.zeros(mesh.n_points, dtype=bool)
    is_vis[vis_mesh.point_data['orig_idx']] = True
    return is_vis


class ProbeMesh:
    def __init__(
            self,
            mesh: pv.PolyData,
            cam_pos: np.ndarray,
            tgt_pos: np.ndarray,
            index: str | int | None = None,
            img_file: str | Path | None = None,
            orig_mesh_file: Path | None = None,
            cam_view_angle: float = 30,
    ):
        self.mesh = mesh
        self.cam_pos = cam_pos
        self.tgt_pos = tgt_pos
        self.cam_view_angle = cam_view_angle
        self.index = index
        self.img_file = Path(img_file)
        self.orig_mesh_file = orig_mesh_file

    @cached_property
    def img(self) -> PIL.Image.Image:  # noqa
        return PIL.Image.open(self.img_file)    # noqa

    @cached_property
    def orig_mesh(self):
        return pv.read(self.orig_mesh_file)

    @cached_property
    def vertex_visibility(self) -> np.ndarray:
        return calc_visible(mesh=self.mesh, cam=self.camera, res=(1024, 1024), off_screen=True)

    @property
    def camera(self) -> pv.Camera:
        cam = pv.Camera()
        cam.position = self.cam_pos
        cam.focal_point = self.tgt_pos
        cam.SetViewUp(0, 0, 1)
        cam.view_angle = self.cam_view_angle
        return cam

    def mesh_to_camera(self):
        t = pv.array_from_vtkmatrix(self.camera.GetViewTransformMatrix())
        return self.mesh.transform(t, inplace=False)

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
    def load_ga_stim(data_dir: Path, scenes_df: pd.DataFrame, leave_pbar=False):
        for scene_id, r in tqdm(scenes_df.iterrows(), total=len(scenes_df), leave=leave_pbar):
            orig = pv.read(data_dir / r.remeshed)
            from_cam_transform = np.linalg.inv(orig.field_data['to_cam_transform'])
            mesh = pv.read(data_dir / r.simplified).transform(from_cam_transform, inplace=False)

            pm = ProbeMesh(
                mesh=mesh,
                cam_pos=orig.field_data['cam_pos'],
                tgt_pos=orig.field_data['cam_focal_point'],
                index=int(scene_id),
                img_file=data_dir / r.render,
            )
            yield pm
            
    def plot_weights(
            self,
            weights: np.ndarray,
            shape: tuple[int, int] | None = None,
            link_views=True,
            render=True,
            scalar_bar=True,
            titles=None,
            **kwargs
    ):
        if shape is None:
            if weights.ndim > 1:
                n = int(np.ceil(np.sqrt(weights.shape[1])))
                shape = (n, n)

        plotter = pv.Plotter(shape=shape, off_screen=render, **kwargs)

        if weights.ndim == 1:
            weights = weights.reshape(1, -1)
            plotters = ((0, 0), plotter),
        else:
            plotters = iter_subplots(plotter, shape=shape, link_views=link_views)

        if titles is None:
            titles = ['' for _ in range(np.prod(shape))]

        for i, ((row_col, p), ttl) in enumerate(zip(plotters, titles)):
            p.add_mesh(self.mesh.copy(), scalars=weights[:, i], show_scalar_bar=i == 0 and scalar_bar)
            p.camera = self.camera

            if ttl:
                p.add_title(ttl)

        if render:
            return PIL.Image.fromarray(plotter.screenshot())  # noqa
        else:
            plotter.show()
            return plotter

    def render(
            self,
            window_size=None,
            ground=True,
            weights=None,
            weights_clim_pctile: tuple[float, float] | None = None,
            **kwargs,
    ):
        mesh = self.mesh
        if weights is not None:
            mesh = mesh.copy()
            mesh.point_data['weights'] = weights
            kwargs['scalars'] = 'weights'

            if weights_clim_pctile:
                w = weights[self.vertex_visibility]
                w0, w1 = np.percentile(w, weights_clim_pctile)
                kwargs['clim'] = (w0, w1)

        p = pv.Plotter(window_size=window_size, off_screen=True)
        p.add_mesh(mesh, **kwargs)
        p.camera = self.camera
        if ground:
            p.add_mesh(pv.Plane(), color='gray')

        img = p.screenshot()
        p.close()
        return PIL.Image.fromarray(img)  # noqa

    def render_weights(self, weights: np.ndarray, img_sz=(500, 500)):
        from torchvision.transforms.functional import to_tensor  # noqa

        imgs = [
            to_tensor(
                self.render(
                    window_size=img_sz, ground=False, scalars=w, show_scalar_bar=False
                )
            )
            for w in tqdm(weights.T, total=weights.shape[1])
        ]
        return imgs

@dataclass
class VertexWeights:
    weights: np.ndarray  # (n_verts, n_channels)

    def pca(self, n=3, normalize=True):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n)
        w_hat = pca.fit_transform(self.weights)

        if normalize:
            w_hat_min = w_hat.min(axis=0)
            w_hat_max = w_hat.max(axis=0)
            w_hat_norm = (w_hat - w_hat_min) / (w_hat_max - w_hat_min)
            w_hat_norm = (w_hat_norm * 2) - 1
            return w_hat_norm
        else:
            return w_hat


@dataclass
class ProbeMeshSpec:
    mesh_file: str | Path
    cam_pos: tuple[float, float, float]
    tgt_pos: tuple[float, float, float]
    cam_view_angle: float
    repair: bool = True
    fill_holes: bool = False
    index: str | None = None
    wrap: bool = True
    relative_alpha: float = 100
    relative_offset: float = 600
    simplify_n_faces: int | None = None
    img_file: Path | str | None = None

    def _probe_mesh(self, probe_mesh: pv.PolyData) -> ProbeMesh:
        return ProbeMesh(
            mesh=probe_mesh,
            cam_pos=np.array(self.cam_pos),
            tgt_pos=np.array(self.tgt_pos),
            cam_view_angle=self.cam_view_angle,
            index=self.index,
            img_file=self.img_file,
            orig_mesh_file=Path(self.mesh_file),
        )

    def cached(self, verbose: bool = False, recache: bool = False) -> ProbeMesh:
        f = Path(self.mesh_file)
        cache_file = f.with_suffix('.cache' + f.suffix)

        if not recache and cache_file.exists():
            return self._probe_mesh(pv.read(cache_file))
        else:
            pm = self.load(verbose=verbose)
            pm.mesh.save(cache_file)
            return pm

    def load(self, verbose: bool = False) -> ProbeMesh:
        import pymeshfix  # noqa
        from seagullmesh import Mesh3

        if verbose:
            def _log(x):
                print(x)
        else:
            def _log(x):
                pass

        mesh_file = Path(self.mesh_file)

        _log(f"Loading file {mesh_file}")
        pv_mesh = pv.read(mesh_file).triangulate().clean()

        if self.repair:
            _log('pymeshfix')
            meshfix = pymeshfix.MeshFix(pv_mesh.points, pv_mesh.regular_faces)
            meshfix.repair()
            pv_mesh = meshfix.mesh

        try:
            sm_mesh = Mesh3.from_pyvista(pv_mesh)
        except RuntimeError as e:
            _log(f"Error: {type(e)}: {e}")
            if (str(e) != 'Polygon orientation failed') or not self.wrap:
                raise
            _log("Alpha wrapping triangles")
            sm_mesh = Mesh3.from_alpha_wrapping(
                points=pv_mesh.points,
                faces=pv_mesh.regular_faces,
                relative_alpha=self.relative_alpha,
                relative_offset=self.relative_offset,
            )
        else:
            if self.fill_holes:
                _log('fill holes')
                sm_mesh.triangulate_holes(sm_mesh.extract_boundary_cycles())

            if self.wrap:
                _log('alpha wrapping')
                sm_mesh = sm_mesh.alpha_wrapping(
                    relative_alpha=self.relative_alpha,
                    relative_offset=self.relative_offset,
                )

        if self.simplify_n_faces:
            _log('edge collapse')
            sm_mesh.edge_collapse('face', self.simplify_n_faces)

        return self._probe_mesh(sm_mesh.to_pyvista())
    
    @staticmethod
    def defined() -> Iterator[ProbeMeshSpec]:
        yield ProbeMeshSpec(
            mesh_file=r"D:\mesh_extract_work\VervertMonkey\JF0N2N0B_VervetMonkeys_Blender\JF0N2N0B_VervetMonkeys_Blender\JF0N2N0B_VervetMonkeys_Run_v4_mesh_extraction.ply",
            img_file=r"C:\corpus2_temp\JF0N2N0B_VervetMonkeys_Run_v4_orig_Main.png",
            index='JF0N2N0B_VervetMonkeys_Run_v4_orig_Main.png',
            cam_pos=(-0.883204, -1.11311, 1.03748),
            tgt_pos=(0.062604, 0.152099, 0.187549),
            cam_view_angle=39.6,
            repair=True,
            fill_holes=True,
            wrap=True,
            relative_alpha=600,
            relative_offset=1000,
            simplify_n_faces=3000,

        )
    
        yield ProbeMeshSpec(
            mesh_file=r"D:\mesh_extract_work\GreyTabbyCat\JC0L219A2_GreyTabbyCat_Blender\JC0L219A2_GreyTabbyCat_Blender\JC0L219A2_GreyTabbyCat_Trot_v4.mesh_extraction2.ply",
            img_file=r"C:\corpus2_temp\JC0L219A2_GreyTabbyCat_Trot_v4_orig_Main.png",
            index='JC0L219A2_GreyTabbyCat_Trot_v4_orig_Main.png',
            cam_pos=(-0.19411747, -0.86916119, 0.33897164),
            tgt_pos=(2.28889287e-04, -3.71107072e-01, 1.14519626e-01),
            cam_view_angle=39.6,
            repair=True,
            fill_holes=True,
            wrap=True,
            relative_alpha=100,
            relative_offset=600,
            simplify_n_faces=3000,
        )
    
        yield ProbeMeshSpec(
            mesh_file=r"D:\mesh_extract_work\SnowshoeHare\JU0L219A1_SnowshoeHare_Summer_Blender\JU0L219A1_SnowshoeHare_Summer_Jump_v4_A_mesh_extraction.ply",
            index='JU0L219A1_SnowshoeHare_Summer_Jump_v4_A_orig_Main.png',
            img_file=r"C:\corpus2_temp\JU0L219A1_SnowshoeHare_Summer_Jump_v4_A_orig_Main.png",
            cam_pos=(-0.52659595, -0.80872011, 0.40757233),
            tgt_pos=(0.019130593165755272, -0.4574986398220062, 0.13628436625003815),
            cam_view_angle=39.6,
            repair=True,
            fill_holes=True,
            wrap=True,
            relative_alpha=600,
            relative_offset=1000,
            simplify_n_faces=3000,
        )
    
        yield ProbeMeshSpec(
            mesh_file=r"D:\mesh_extract_work\black rattlesnake\BlackRatSnake_Rigged_blender\BlackRatSnake_FK_Poses_v4_mesh_extraction.ply",
            index='BlackRatSnake_FK_Poses_v4_orig_Main.png',
            img_file=r"C:\corpus2_temp\BlackRatSnake_FK_Poses_v4_orig_Main.png",
            cam_pos=(-0.7484257221221924, 0.3798489570617676, 0.4411892890930176),
            tgt_pos=(-0.003663875162601471, -0.04849287122488022, -0.004026277922093868),
            cam_view_angle=39.6,
            repair=True,
            fill_holes=True,
            wrap=True,
            relative_alpha=600,
            relative_offset=1000,
            simplify_n_faces=3000,
        )
        
        yield ProbeMeshSpec(
            mesh_file=r"D:\mesh_extract_work\Cicada\Cicada_Crawl_v3_mesh_extraction.ply",
            index='Cicada_Crawl_v3_orig.png',
            img_file=r"C:\corpus2_temp\Cicada_Crawl_v3_orig.png",
            cam_pos=(1.7629317045211792, -0.2856280505657196, 0.9172284603118896),
            tgt_pos=(0.05397617816925049, 0.12930458784103394, 0.15642812848091125),
            cam_view_angle=39.6,
            repair=True,
            fill_holes=True,
            wrap=True,
            relative_alpha=600,
            relative_offset=1000,
            simplify_n_faces=3000,
        )
    
        yield ProbeMeshSpec(
            mesh_file=r"D:\mesh_extract_work\frog\frog_v4_mesh_extraction.ply",
            index='frog_v4_orig_Main.png',
            img_file=r"C:\corpus2_temp\frog_v4_orig_Main.png",
            cam_pos=(0.14727137982845306, -0.2970578372478485, 0.3091987669467926),
            tgt_pos=(0.01088304165750742, 0.0, 0.04733939468860626),
            cam_view_angle=35.49,
            repair=True,
            fill_holes=True,
            wrap=True,
            relative_alpha=600,
            relative_offset=1000,
            simplify_n_faces=3000,
        )
    
        yield ProbeMeshSpec(
            mesh_file=r"D:\mesh_extract_work\SpottedTurtle\Spotted-turtle_v4_mesh_extraction.ply",
            index='Spotted-turtle_v4_orig_Main.png',
            img_file=r"C:\corpus2_temp\Spotted-turtle_v4_orig_Main.png",
            cam_pos=(0.2038934826850891, -0.17945489287376404, 0.20085172355175018),
            tgt_pos=(0.0009141467162407935, 0.0057134167291224, 0.005089262500405312),
            cam_view_angle=39.6,
            repair=True,
            fill_holes=True,
            wrap=True,
            relative_alpha=600,
            relative_offset=1000,
            simplify_n_faces=3000,
        )
    
        yield ProbeMeshSpec(
            mesh_file=r"D:\mesh_extract_work\Squirrel\Squirrel_v4_mesh_extraction.ply",
            index='Squirrel_v4_orig_Main.png',
            img_file=r"C:\corpus2_temp\Squirrel_v4_orig_Main.png",
            cam_pos=(10.23602294921875, -4.244842052459717, 9.324092864990234),
            tgt_pos=(-0.02328479290008545, -2.0086560249328613, 1.9039499759674072),
            cam_view_angle=49.13,
            repair=True,
            fill_holes=True,
            wrap=True,
            relative_alpha=600,
            relative_offset=1000,
            simplify_n_faces=3000,
        )
    
        yield ProbeMeshSpec(
            mesh_file=r"D:\mesh_extract_work\chipmunk\JF0N2N0A_Chipmunk_Blender\JF0N2N0A_Chipmunk_FindFood_v4_mesh_extraction.ply",
            index='JF0N2N0A_Chipmunk_FindFood_v4_orig_Main.png',
            img_file=r"C:\corpus2_temp\JF0N2N0A_Chipmunk_FindFood_v4_orig_Main.png",
            cam_pos=(-0.09019485861063004, -0.5623771548271179, 0.13063561916351318),
            tgt_pos=(0.002574823796749115, -0.3826358914375305, 0.031962279230356216),
            cam_view_angle=39.6,
            repair=True,
            fill_holes=True,
            wrap=True,
            relative_alpha=600,
            relative_offset=1000,
            simplify_n_faces=3000,
        )
    
        yield ProbeMeshSpec(
            mesh_file=r"D:\mesh_extract_work\barcelona door handle brass\Barcelona Door Handle Brass_wip_v3.ply",
            index='Barcelona Door Handle Brass_wip_v3_orig.png',
            img_file=r"C:\corpus2_temp\Barcelona Door Handle Brass_wip_v3_orig.png",
            cam_pos=(-0.038916219025850296, -0.17309686541557312, 0.23030444979667664),
            tgt_pos=(-0.002232855651527643, 0.011647812090814114, 0.013855653814971447),
            cam_view_angle=39.6,
            repair=False,
            fill_holes=False,
            wrap=True,
            relative_alpha=600,
            relative_offset=1000,
            simplify_n_faces=3000,
        )
    
        yield ProbeMeshSpec(
            mesh_file=r"D:\mesh_extract_work\door knocker\Vintage_Ring_Shaped_Door_Knocker_v3_mesh_extraction.ply",
            index='Vintage_Ring_Shaped_Door_Knocker_v3_orig.png',
            img_file=r"C:\corpus2_temp\Vintage_Ring_Shaped_Door_Knocker_v3_orig.png",
            cam_pos=(18.830228805541992, -1.0160112651647069e-05, 9.02007007598877),
            tgt_pos=(0.0, 0.0, 5.857663154602051),
            cam_view_angle=39.6,
            repair=False,
            fill_holes=False,
            wrap=True,
            relative_alpha=600,
            relative_offset=1000,
            simplify_n_faces=3000,
        )
    
        yield ProbeMeshSpec(
            mesh_file=r"D:\mesh_extract_work\doorknob\door_knob_v3_mesh_extraction.ply",
            index='door_knob_v3_orig.png',
            img_file=r"C:\corpus2_temp\door_knob_v3_orig.png",
            cam_pos=(-2.720917224884033, -1.9613797664642334, 3.114717960357666),
            tgt_pos=(0.0, 0.0, 0.8159806132316589),
            cam_view_angle=49.13,
            repair=False,
            fill_holes=False,
            wrap=True,
            relative_alpha=600,
            relative_offset=1000,
            simplify_n_faces=3000,
        )
    
        yield ProbeMeshSpec(
            mesh_file=r"C:\Users\dg\OneDrive\Documents\experiment\turbosquidfiles\floor lamp\FLOOR_LAMP_02_v3_mesh_extraction.ply",
            index='FLOOR_LAMP_02_v3_orig.png',
            img_file=r"C:\corpus2_temp\FLOOR_LAMP_02_v3_orig.png",
            cam_pos=(16.267391204833984, 0.5502005815505981, 13.235892295837402),
            tgt_pos=(0.0, 1.664716362953186, 5.1058759689331055),
            cam_view_angle=39.6,
            repair=False,
            fill_holes=False,
            wrap=True,
            relative_alpha=600,
            relative_offset=1000,
            simplify_n_faces=3000,
        )
    
        yield ProbeMeshSpec(
            mesh_file=r"D:\mesh_extract_work\vintage lamp\FLOOR_LAMP_03_v3_mesh_extraction.ply",
            index='FLOOR_LAMP_03_v3_orig.png',
            img_file=r"C:\corpus2_temp\FLOOR_LAMP_03_v3_orig.png",
            cam_pos=(8.832108497619629, -1.5337945222854614, 6.305342197418213),
            tgt_pos=(0.0, 0.0, 2.986098051071167),
            cam_view_angle=39.6,
            repair=False,
            fill_holes=False,
            wrap=True,
            relative_alpha=600,
            relative_offset=1000,
            simplify_n_faces=3000,
        )
    
        yield ProbeMeshSpec(
            mesh_file=r"D:\mesh_extract_work\lamp\Lamp_v3_mesh_extraction.ply",
            index='Lamp_v3_orig.png',
            img_file=r"C:\corpus2_temp\Lamp_v3_orig.png",
            cam_pos=(-0.8790243864059448, 0.008201121352612972, 0.613591730594635),
            tgt_pos=(0.011321105062961578, -0.08542653918266296, 0.2687967121601105),
            cam_view_angle=39.6,
            repair=False,
            fill_holes=False,
            wrap=True,
            relative_alpha=600,
            relative_offset=1000,
            simplify_n_faces=3000,
        )
    
        yield ProbeMeshSpec(
            mesh_file=r"D:\mesh_extract_work\muuto light\Muuto_Tip_Table_Light_v3_mesh_extraction.ply",
            index='Muuto_Tip_Table_Light_v3_orig.png',
            img_file=r"C:\corpus2_temp\Muuto_Tip_Table_Light_v3_orig.png",
            cam_pos=(2.520118474960327, 22.443445205688477, 15.19332504272461),
            tgt_pos=(0.08447802066802979, 0.232452392578125, 6.148859977722168),
            cam_view_angle=39.6,
            repair=False,
            fill_holes=False,
            wrap=True,
            relative_alpha=600,
            relative_offset=1000,
            simplify_n_faces=3000,
        )
    
        yield ProbeMeshSpec(
            mesh_file=r"D:\mesh_extract_work\return.handle\Door Handle 015 v3_mesh_extraction.ply",
            index='Door Handle 015 v3_orig.png',
            img_file=r"C:\corpus2_temp\Door Handle 015 v3_orig.png",
            cam_pos=(0.07338540256023407, -0.20358070731163025, 0.21148289740085602),
            tgt_pos=(0.058274075388908386, 0.0, 0.02320905774831772),
            cam_view_angle=39.6,
            repair=False,
            fill_holes=False,
            wrap=True,
            relative_alpha=600,
            relative_offset=600,
            simplify_n_faces=3000,
        )
    
        yield ProbeMeshSpec(
            mesh_file=r"D:\mesh_extract_work\liquor\liquor_bottles_v4_mesh_extraction_ballantine.ply",
            index='liquor_bottles_v4_orig_Ballantine.png',
            img_file=r"C:\corpus2_temp\liquor_bottles_v4_orig_Ballantine.png",
            cam_pos=(0.09443426877260208, -1.611390233039856, 0.814530611038208),
            tgt_pos=(0.0, 0.0, 0.09535327553749084),
            cam_view_angle=20.41,
            repair=False,
            fill_holes=False,
            wrap=True,
            relative_alpha=600,
            relative_offset=600,
            simplify_n_faces=3000,
        )
    
        yield ProbeMeshSpec(
            mesh_file=r"D:\mesh_extract_work\liquor\liquor_bottles_v4_mesh_extraction_bombay.ply",
            index='liquor_bottles_v4_orig_Bombay.png',
            img_file=r"C:\corpus2_temp\liquor_bottles_v4_orig_Bombay.png",
            cam_pos=(0.09443426877260208, -1.611390233039856, 0.814530611038208),
            tgt_pos=(0.0, 0.0, 0.09535327553749084),
            cam_view_angle=20.41,
            repair=False,
            fill_holes=False,
            wrap=True,
            relative_alpha=600,
            relative_offset=600,
            simplify_n_faces=3000,
        )
    
        yield ProbeMeshSpec(
            mesh_file=r"D:\mesh_extract_work\Pigeon\JF0L421A0_Pigeon_Blender\JF0L421A0_Pigeon_Blender\JF0L421A0_Pigeon_Land_v4_mesh_extraction.ply",
            index='JF0L421A0_Pigeon_Land_v4_orig_Main.png',
            img_file=r"C:\corpus2_temp\JF0L421A0_Pigeon_Land_v4_orig_Main.png",
            cam_pos=(0.4658816456794739, -0.3691014051437378, 0.5186117887496948),
            tgt_pos=(0.2733516991138458, 0.28120970726013184, 0.1614539474248886),
            cam_view_angle=39.6,
            repair=False,
            fill_holes=False,
            wrap=True,
            relative_alpha=600,
            relative_offset=1000,
            simplify_n_faces=5000,
        )
    
        yield ProbeMeshSpec(
            mesh_file=r"D:\mesh_extract_work\robin\robin_animated_v4_mesh_extraction.ply",
            index='robin_animated_v4_orig_Main.png',
            img_file=r"C:\corpus2_temp\robin_animated_v4_orig_Main.png",
            cam_pos=(-31.297409057617188, -16.788650512695312, 8.073495864868164),
            tgt_pos=(-0.15069210529327393, -2.3357276916503906, 4.40774393081665),
            cam_view_angle=35.49,
            repair=False,
            fill_holes=False,
            wrap=True,
            relative_alpha=600,
            relative_offset=1000,
            simplify_n_faces=5000,
        )
    
        yield ProbeMeshSpec(
            mesh_file=r"D:\mesh_extract_work\nespr4esso\Coffee_Machine_v3_mesh_extraction.ply",
            index='Coffee_Machine_v3_orig.png',
            img_file=r"C:\corpus2_temp\Coffee_Machine_v3_orig.png",
            cam_pos=(0.4897646903991699, 1.369809627532959, 1.192300796508789),
            tgt_pos=(0.039121244102716446, 0.007926076650619507, 0.2581562101840973),
            cam_view_angle=21.03,
            repair=False,
            fill_holes=False,
            wrap=True,
            relative_alpha=600,
            relative_offset=1000,
            simplify_n_faces=3000,
        )

        yield ProbeMeshSpec(
            mesh_file=r"D:\mesh_extract_work\Lemon\Lemon_v3_mesh_extraction.ply",
            index='Lemon_v3_orig.png',
            img_file=r"D:\corpus2_temp\Lemon_v3_orig.png",
            cam_pos=(0.0, -0.31430575251579285, 0.14941911399364471),
            tgt_pos=(0.0, 0.0, 0.0224834643304348),
            cam_view_angle=23.91,
            repair=False,
            fill_holes=False,
            wrap=True,
            relative_alpha=600,
            relative_offset=1000,
            simplify_n_faces=3000,
        )

        yield ProbeMeshSpec(
            mesh_file=r"D:\mesh_extract_work\Mouse with mousepad\Mouse with Mousepad_v3_mesh_extraction.ply",
            index='Mouse with Mousepad_v3_orig.png',
            img_file=r"D:\corpus2_temp\Mouse with Mousepad_v3_orig.png",
            cam_pos=(0.0, -0.4066583514213562, 0.4414394795894623),
            tgt_pos=(0.022418677806854248, 0.0, 0.007812324911355972),
            cam_view_angle=20.41,
            repair=False,
            fill_holes=False,
            wrap=True,
            relative_alpha=600,
            relative_offset=1000,
            simplify_n_faces=3000,
        )