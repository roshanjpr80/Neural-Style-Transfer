import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image, ImageFile
from torchvision import transforms
from torchvision.utils import save_image as _tv_save_image

# Large scraped datasets (Painter by Numbers, COCO) occasionally contain
# truncated JPEGs. Without this, PIL raises on the first one and kills a
# training run that may have been going for hours.
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')


class ImageFolderDataset(Dataset):
    """
    Loads every image directly inside `root` (optionally its subfolders too),
    with no labels — the right shape for NST training, where content and
    style images are sampled independently rather than paired.

    Parameters
    ----------
    root : str or Path
        Folder containing images.
    transform : callable, optional
        Usually the output of get_transform().
    recursive : bool, default False
        If True, also scans subfolders (e.g. class-organised folders, or a
        Kaggle download that unzipped into nested directories).
    validate : bool, default False
        If True, opens every image once at init time and silently drops any
        that fail to load, so a bad file surfaces as a smaller dataset
        instead of a crash midway through training. Slower to start up on
        large datasets, so it's off by default — turn it on the first time
        you point this at a new, unfamiliar folder (like a fresh Painter by
        Numbers or COCO download), then you can leave it off for reruns.
    """

    def __init__(self, root, transform=None, recursive=False, validate=False):
        super(ImageFolderDataset, self).__init__()
        self.root = Path(root)
        self.transform = transform

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset folder not found: {self.root}")

        if recursive:
            candidates = [p for p in self.root.rglob('*') if p.is_file()]
        else:
            candidates = [self.root / f for f in os.listdir(self.root)]

        self.files = [
            p for p in candidates
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if len(self.files) == 0:
            raise RuntimeError(f"No images found in dataset folder: {self.root}")

        if validate:
            good_files = []
            skipped = 0
            for path in self.files:
                try:
                    with Image.open(path) as img:
                        img.verify()
                    good_files.append(path)
                except Exception:
                    skipped += 1
            self.files = good_files
            if skipped:
                print(f"[ImageFolderDataset] Skipped {skipped} unreadable file(s) in {self.root}.")
            if len(self.files) == 0:
                raise RuntimeError(f"All files in {self.root} failed validation.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = self.files[idx]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            # A single corrupted file shouldn't kill a multi-hour training run.
            # Fall back to a different random image instead of crashing.
            print(f"[ImageFolderDataset] Skipping unreadable file {image_path}: {e}")
            return self[(idx + 1) % len(self)]

        if self.transform:
            image = self.transform(image)

        return image


def get_transform(size, crop, final_size):
    """
    Standard NST preprocessing pipeline.

    Note: do NOT add torchvision.transforms.Normalize with ImageNet mean/std
    here. VGGEncoder's first layer (a 1x1 conv, see models.py) already bakes
    that normalisation into the network's own trained weights — normalising
    twice will feed the encoder out-of-distribution values.
    """
    transform_list = []
    if size > 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.RandomCrop(final_size))
    else:
        transform_list.append(transforms.Resize(final_size))

    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


class InfiniteSampler(Sampler):
    """
    Wraps a dataset so it can be iterated forever, reshuffling each time it
    runs out. NST training is normally driven by a fixed iteration count
    (e.g. 160,000 steps), not epochs, since content and style datasets are
    unrelated in size and don't need to be "completed" together — this makes
    a DataLoader built on a small style dataset behave like an endless stream.
    """

    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(self._infinite_shuffle())

    def __len__(self):
        return 2 ** 31  # effectively infinite for DataLoader's bookkeeping

    def _infinite_shuffle(self):
        while True:
            order = torch.randperm(self.num_samples).tolist()
            for i in order:
                yield i


def get_dataloader(root, batch_size, size=512, crop=True, final_size=256,
                    recursive=False, validate=False, num_workers=4, infinite=True):
    """
    Convenience wrapper: folder path in, ready-to-iterate DataLoader out.

    Example
    -------
    >>> content_loader = get_dataloader("data/coco/train2017", batch_size=8)
    >>> style_loader = get_dataloader("data/painter_by_numbers/train_1", batch_size=8)
    >>> content_iter, style_iter = iter(content_loader), iter(style_loader)
    >>> content_batch = next(content_iter)
    >>> style_batch = next(style_iter)
    """
    transform = get_transform(size, crop, final_size)
    dataset = ImageFolderDataset(root, transform=transform, recursive=recursive, validate=validate)

    sampler = InfiniteSampler(dataset) if infinite else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )


def save_output_image(tensor, path):
    """
    Save a (1, 3, H, W) or (3, H, W) tensor in [0, 1] range as an image file.
    Useful for writing out training-checkpoint previews or stylize() results
    from models.py's StyleTransferModel without pulling in extra dependencies.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _tv_save_image(tensor.clamp(0, 1), str(path))


def calc_mean_std(feat, eps=1e-5):
    """
    Per-channel mean and standard deviation of a (N, C, H, W) feature map.
    Uses the biased (population) variance estimator, matching the original
    AdaIN paper's implementation exactly — keep this the single source of
    truth for this calculation rather than redefining it elsewhere.
    """
    size = feat.size()
    assert (len(size) == 4)
    batch_size, channels = size[:2]
    feat_mean = feat.view(batch_size, channels, -1).mean(dim=2).view(batch_size, channels, 1, 1)
    feat_var = feat.view(batch_size, channels, -1).var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(batch_size, channels, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    # [batch size, channels, h, w]
    assert content_feat.size()[:2] == style_feat.size()[:2], (
        "Content and style features must share the same batch size "
        "and channel count."
    )
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_content_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return (
        normalized_content_feat * style_std.expand(size)
        + style_mean.expand(size)
    )