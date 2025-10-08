from pathlib import Path
from typing import cast, Iterator

import numpy as np
import pyvista as pv
from seagullmesh import Mesh3
import pandas as pd
from tqdm import tqdm

from ga_regression import specs

spec = specs(root=r"/home/darik/resynth")[9]

file = spec.data_file
root = file.parent
scenes = cast(pd.DataFrame, pd.read_hdf(file, key='scenes'))
n_faces = [500,]
n_total = len(scenes)
if n_faces:
    n_total *= len(n_faces)


shuffle_visible = True
shuffle_colors = False


def iter_files() -> Iterator[tuple[Path, Path, Path]]:
    for orig, simplified, md in tqdm(zip(scenes.remeshed, scenes.simplified, scenes.mesh_data), total=n_total):
        if not orig or not simplified or not md:
            continue

        f_orig = root / orig
        f_simp = root / simplified
        f_md = root / md

        if n_faces:
            for n in n_faces:
                yield f_orig, f_simp.with_suffix(f'.{n}_faces' + f_simp.suffix), f_md
        else:
            yield f_orig, f_simp, f_md


def process():
    for f_orig, f_simp, f_md in tqdm(iter_files(), total=len(scenes)):
        m_simp = cast(pv.PolyData, pv.read(f_simp))

        if shuffle_visible:
            shuffled_vis = np.random.permutation(m_simp.point_data['visible'])
            outfile = f_simp.with_suffix(f_simp.suffix + '.shuffled_visible.npy')
            np.save(outfile, shuffled_vis)

        if shuffle_colors:
            m_orig = cast(pv.PolyData, pv.read(f_orig))
            sm_orig = Mesh3.from_pyvista(m_orig)
            tree = sm_orig.aabb_tree()

            with np.load(f_md) as data:
                colors = data['vert_colors']

            shuffled_colors = colors[np.random.permutation(len(colors)), :]

            sm_simp = Mesh3.from_pyvista(m_simp)
            surf_pts = tree.locate_points(sm_simp.point_soup())
            vert_idxs = surf_pts.faces.triangle_soup()

            m_simp.point_data['color'] = (
                    colors[vert_idxs] * surf_pts.bary_coords[:, :, np.newaxis]).sum(axis=1)
            m_simp.point_data['shuffled_color'] = (
                    shuffled_colors[vert_idxs] * surf_pts.bary_coords[:, :, np.newaxis]).sum(axis=1)


if __name__ == '__main__':
    process()




