import os
from typing import Iterable, Tuple
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader, random_split
from pose_model.architecture.preprocess import Preprocessor
from pose_model.utils import load_png, image_to_tensor


class AssetsDataset(IterableDataset):
    def __init__(self, assets_dir: Path):
        self.assets_dir = assets_dir
        self.preprocessor = Preprocessor()
    
    def _asset_files_iter(self) -> Iterable[Tuple[Path, Path]]:
        for char_dir in self.assets_dir.iterdir():
            for anim_dir in char_dir.iterdir():
                frames_dir = anim_dir / 'frames'
                poses_dir = anim_dir / 'poses'
                reference_file = anim_dir / 'reference.png'
                if frames_dir.exists() and poses_dir.exists() and reference_file.exists():
                    for target_file, pose_file in zip(frames_dir.iterdir(), poses_dir.iterdir()):
                        yield reference_file, pose_file, target_file

    def __iter__(self):
        for ref_file, pose_file, target_file in self._asset_files_iter():
            # Load images
            ref_img = load_png(ref_file)
            pose_img = load_png(pose_file)
            target_img = load_png(target_file)

            # Preprocess
            latent, clip  = self.preprocessor.preprocess(ref_img)
            target = image_to_tensor(target_img)
            pose = image_to_tensor(pose_img)

            yield {
                'ref_img': ref_img,
                'pose_img': pose_img,
                'target_img': target_img,
                'ref_latent': latent,
                'ref_clip': clip,
                'pose': pose,
                'target': target
            }


def compile_dataset(dataset: Dataset, out_dir: Path, samples_per_file: 1000,
                    show_progress: bool = True):
    os.makedirs(out_dir, exist_okay=True)
    dl = DataLoader(dataset)

    buffer = []
    with tqdm(disable=not show_progress) as pbar:
        for i, data in enumerate(dl):
            buffer.append(data)
            if len(buffer) == samples_per_file:
                torch.save(buffer, out_dir / f'{i//samples_per_file:04d}.pt')

                # Reset buffer
                buffer = []
                torch.cuda.empty_cache()

            pbar.update(1)
        
        # Save any leftover buffer contents
        if len(buffer) > 0:
            torch.save(buffer, out_dir / f'{i//samples_per_file:04d}.pt')

def count_samples(file: Path) -> int:
    samples = torch.load(str(file))
    count = len(samples)
    del samples # Frees memory
    return count

class PreprocessedAssetsDataset(IterableDataset):
    def __init__(self, assets_dir: Path, map_location: str='cpu'):
        self.assets_dir = assets_dir
        self.map_location = map_location
        self._files = self.assets_dir.glob('*.pt')
        self._file_sizes = [count_samples(f) for f in self._files]

    def __len__(self) -> int:
        return sum(self._file_sizes)

    def __iter__(self):
        for file in self.assets_dir.glob('*.pt'):
            samples = torch.load(file, map_location=self.map_location)
            for samp in samples:
                yield samp


def load_and_split_preprocessed_dataset(assets_dir: Path,
                                        batch_size: int,
                                        train_size: float=0.8,
                                        map_location: str='cpu'
                                        ) -> Tuple[DataLoader, DataLoader]:
    if not assets_dir.exists():
        raise FileNotFoundError(f"assets_dir {assets_dir} does not exist")
    
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0")
    
    if train_size <= 0.0 or train_size >= 1.0:
        raise ValueError(f"train_size must be > 0.0 and < 1.0")

    ds = PreprocessedAssetsDataset(assets_dir, map_location=map_location)

    n_total = len(ds)
    n_train = train_size * n_total
    n_eval = n_train - n_total

    train_ds, eval_ds = random_split(ds, [n_train, n_eval])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=8, pin_memory=True)
    eval_dl = DataLoader(eval_ds, batch_size=batch_size, num_workers=8)

    return train_dl, eval_dl
