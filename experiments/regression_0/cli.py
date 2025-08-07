import argparse

from ga_dataset import *
from ga_regression import *   # Need to import everything for unpickling to work I think?


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path)
    parser.add_argument("--model_idx", type=int)
    parser.add_argument("--stimuli", type=Path)
    parser.add_argument("--outfile", type=Path)
    parser.add_argument("--outkey", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    readers = Readers.from_file(args.models)
    reader = readers[args.model_idx]
    meta = reader.metadata
    stim_file = Path(args.stimuli)
    op_cache = stim_file.parent / 'op_cache'
    op_cache.mkdir(exist_ok=True)

    df_scenes = NeurophysData.load_scenes(
        data_file=stim_file,
        file_mode=meta.opts.mesh_file_mode,
        n_faces=meta.n_faces,
        features=meta.input_features,
    )
    n_scenes = len(df_scenes)
    n_channel = len(meta.channel)

    df_responses = pd.DataFrame(
        np.zeros((n_scenes, n_channel)),
        index=df_scenes.index,
        columns=pd.Index(meta.channel, name='channel'),
    )

    dataset = GaDataset(
        df=df_scenes,
        responses=df_responses,
        root_dir=stim_file.parent,
        k_eig=meta.k_eig,
        op_cache_dir=op_cache,
        file_mode=meta.opts.mesh_file_mode,
        weights=None,
        use_visible=meta.use_visible,
        use_color=meta.use_color,
        norm_verts=meta.norm_verts,
        features=meta.input_features,
        augment=None,
    )

    loader = DataLoader(dataset, shuffle=False, batch_size=None)
    expt = reader.experiment()

    expt.model.eval()
    preds = []
    dummy_preds = np.full(len(meta.channel), np.nan)

    with torch.no_grad():
        for i in range(len(df_scenes)):
            try:
                mesh_data = dataset[i]
            except TypeError as e:
                logger.error("Error loading scene %s: %s", df_scenes.index[i], str(e))
                preds_i = dummy_preds
            else:
                _, preds_i, _ = expt.load_item(mesh_data)
            preds.append(preds_i.cpu().numpy())

    preds = np.stack(preds)
    df_preds = pd.DataFrame(preds, index=df_scenes.index, columns=pd.Index(np.arange(n_channel), name='channel_idx'))
    df_preds.to_hdf(args.outfile, key=args.outkey)
