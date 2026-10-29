import pickle
import random
import sys

import cv2
import numpy as np
import torch
import torch.utils.data as data
from natsort import natsorted
from pathlib import Path

try:
    sys.path.append("..")
    import data.util as util
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def gaus2d(x, y, mx, my, sx, sy):
    """Normalized 2-D Gaussian evaluated on meshgrid arrays."""
    return (
        1.0 / (2.0 * np.pi * sx * sy)
        * np.exp(-((x - mx) ** 2.0 / (2.0 * sx ** 2.0)
                   + (y - my) ** 2.0 / (2.0 * sy ** 2.0)))
    )


def _make_gaussian_vignette(size, mx=0, my=0, sx=60, sy=60):
    """Return a (size, size, 1) Gaussian vignette mask."""
    half = size / 2
    coords = np.linspace(-half, half, size)
    x, y = np.meshgrid(coords, coords)
    return gaus2d(x, y, mx=mx, my=my, sx=sx, sy=sy)[:, :, None]


def _apply_vignette(img, vignette):
    """Multiply *img* by *vignette* and normalize to [0, 1]."""
    img = img * vignette
    return img / np.max(img)


def _to_tensor(img):
    """HWC numpy array ā†' CHW float32 tensor."""
    return torch.from_numpy(
        np.ascontiguousarray(np.transpose(img, (2, 0, 1)))
    ).float()


def _load_split_paths(opt):
    """Load LR / GT path lists from a pickle-based data split."""
    with open(opt["data_split_pkl"], "rb") as f:
        split_list = pickle.load(f)
    if opt["phase"] == "test":
        split_list = natsorted(split_list)
    lr_paths = [f'{opt["dataroot_LQ"]}/{name}.npz' for name in split_list]
    gt_paths = [f'{opt["dataroot_GT"]}/{name}.npz' for name in split_list]
    return lr_paths, gt_paths, split_list


def _assert_paths(lr_paths, gt_paths):
    """Validate that path lists are non-empty and equally sized."""
    assert gt_paths, "Error: GT paths are empty."
    if lr_paths and gt_paths:
        assert len(lr_paths) == len(gt_paths), (
            f"GT and LR datasets have different number of images "
            f"- {len(lr_paths)}, {len(gt_paths)}."
        )


def _build_neighbor_path(base_path, offset, ext, suffix=""):
    """Construct the path for a neighboring frame at *offset* from *base_path*."""
    p = Path(base_path)
    parts = p.stem.split("_")
    # parts layout: <prefix>_<id>_<frame>[_window_<wid>]
    frame_idx = int(parts[2]) + offset
    name = f"{parts[0]}_{parts[1]}_{frame_idx}{suffix}.{ext}"
    return Path(p.parent, name)



class LQGT_1ch_Dataset(data.Dataset):
    """Single-channel LR / GT image pair dataset.

    Set ``opt["vignette"] = True`` to apply a Gaussian vignette mask.
    Subclass and override :meth:`_read_lr_gt` to change how images are loaded.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.LR_env, self.GT_env = "npz", "npz"
        self.LR_size, self.GT_size = opt["LR_size"], opt["GT_size"]
        self.use_vignette = opt.get("vignette", False)

        if opt["data_type"] == "img":
            self.LR_paths, self.GT_paths, self.split_list = _load_split_paths(opt)
        else:
            self.LR_paths, self.GT_paths = None, None
            print("Error: data_type is not matched in Dataset")

        _assert_paths(self.LR_paths, self.GT_paths)

    # ------------------------------------------------------------------
    # Override this in subclasses to change the loading behaviour
    # ------------------------------------------------------------------
    def _read_lr_gt(self, index):
        """Return ``(img_LR, img_GT, LR_path, GT_path)``."""
        GT_path = self.GT_paths[index]
        LR_path = self.LR_paths[index]

        img_GT = util.read_img(self.GT_env, GT_path, None)
        if img_GT.ndim == 2:
            img_GT = np.expand_dims(img_GT, axis=2)

        img_LR = util.read_img(self.LR_env, LR_path, None)
        if img_LR.ndim == 2:
            img_LR = np.expand_dims(img_LR, axis=2)

        return img_LR, img_GT, LR_path, GT_path

    # ------------------------------------------------------------------
    def _apply_preprocess(self, img_LR, img_GT):
        """Apply Gaussian vignette only when ``opt['vignette']`` is set."""
        if not self.use_vignette:
            return img_LR, img_GT
        vignette = _make_gaussian_vignette(self.GT_size)
        return _apply_vignette(img_LR, vignette), _apply_vignette(img_GT, vignette)

    def __getitem__(self, index):
        GT_size = self.opt["GT_size"]
        LR_size = self.opt["LR_size"]

        img_LR, img_GT, LR_path, GT_path = self._read_lr_gt(index)

        if self.opt["phase"] == "train":
            H, W, _ = img_LR.shape

            img_LR, img_GT = self._apply_preprocess(img_LR, img_GT)

            # randomly crop
            rnd_h = random.randint(0, max(0, H - LR_size))
            rnd_w = random.randint(0, max(0, W - LR_size))
            img_LR = img_LR[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]
            img_GT = img_GT[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]

            img_LR, img_GT = util.augment(
                [img_LR, img_GT],
                self.opt["use_flip"], self.opt["use_rot"],
                self.opt["mode"], self.opt["use_swap"],
            )

        elif LR_size is not None:
                img_LR, img_GT = self._apply_preprocess(img_LR, img_GT)

        else:
            img_LR, img_GT = self._apply_preprocess(img_LR, img_GT)

        return {
            "LQ": _to_tensor(img_LR),
            "GT": _to_tensor(img_GT),
            "LQ_path": LR_path,
            "GT_path": GT_path,
        }

    def __len__(self):
        return len(self.GT_paths)

class LQGT_5ch_ref_copied_Dataset(LQGT_1ch_Dataset):
    """5-channel LR / GT dataset: center frame + 2 neighbors (copied GT).

    Inherits all training logic (crop, augment, optional vignette) from
    :class:`LQGT_1ch_Dataset` and only overrides the image loading.
    """

    def _read_neighbors(self, lr_path):
        """Read the center frame and its 4 temporal neighbors."""
        read = lambda p: util.read_img(self.LR_env, p, None)
        imgs = [read(_build_neighbor_path(lr_path, off, self.LR_env))
                for off in (-2, -1, 0, 1, 2)]
        # Override center with direct path for safety
        imgs[2] = read(lr_path)
        return np.concatenate(imgs, axis=2)

    def _read_lr_gt(self, index):
        """Return 5-channel ``(img_LR, img_GT, LR_path, GT_path)``."""
        GT_path = self.GT_paths[index]
        LR_path = self.LR_paths[index]

        # GT: replicate single channel 5 times to match LR channel count
        img_GT_single = util.read_img(self.GT_env, GT_path, None)
        img_GT = np.concatenate([img_GT_single] * 5, axis=2)

        img_LR = self._read_neighbors(LR_path)

        return img_LR, img_GT, LR_path, GT_path


class LQGT_5ch_ref_copied_extend_aug_Dataset(LQGT_1ch_Dataset):
    """5-channel LR / GT dataset: center frame + 2 neighbors (copied GT).

    Inherits all training logic (crop, augment, optional vignette) from
    :class:`LQGT_1ch_Dataset` and only overrides the image loading.
    """

    def _read_neighbors(self, lr_path):
        """Read the center frame and its 4 temporal neighbors."""
        read = lambda p: util.read_img(self.LR_env, p, None)
        imgs = [read(_build_neighbor_path(lr_path, off, self.LR_env))
                for off in (-2, -1, 0, 1, 2)]
        # Override center with direct path for safety
        imgs[2] = read(lr_path)
        return np.concatenate(imgs, axis=2)

    def _read_lr_gt(self, index):
        """Return 5-channel ``(img_LR, img_GT, LR_path, GT_path)``."""
        GT_path = self.GT_paths[index]
        LR_path = self.LR_paths[index]

        # GT: replicate single channel 5 times to match LR channel count
        img_GT_single = util.read_img(self.GT_env, GT_path, None)
        img_GT = np.concatenate([img_GT_single] * 5, axis=2)

        img_LR = self._read_neighbors(LR_path)

        return img_LR, img_GT, LR_path, GT_path
    
        # ------------------------------------------------------------------
    def _apply_preprocess(self, img_LR, img_GT):
        """Apply Gaussian vignette only when ``opt['vignette']`` is set."""
        if not self.use_vignette:
            return img_LR, img_GT
        
        sx = random.random()*20 + 40
        sy = random.random()*20 + 40
        vignette = _make_gaussian_vignette(self.GT_size, sx=sx, sy=sy)
        return _apply_vignette(img_LR, vignette), _apply_vignette(img_GT, vignette)


    def __getitem__(self, index):
            GT_size = self.opt["GT_size"]
            LR_size = self.opt["LR_size"]

            img_LR, img_GT, LR_path, GT_path = self._read_lr_gt(index)

            if self.opt["phase"] == "train":
                H, W, _ = img_LR.shape
                
                if random.random() < 0.5:
                    img_LR, img_GT = self._apply_preprocess(img_LR, img_GT)

                # randomly crop
                rnd_h = random.randint(0, max(0, H - LR_size))
                rnd_w = random.randint(0, max(0, W - LR_size))
                img_LR = img_LR[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]
                img_GT = img_GT[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]

                img_LR, img_GT = util.augment(
                    [img_LR, img_GT],
                    self.opt["use_flip"], self.opt["use_rot"],
                    self.opt["mode"], self.opt["use_swap"],
                )
                if random.random() < 0.5:
                    img_LR = (img_LR-img_LR.min()) / (img_LR.max() - img_LR.min())
                    img_GT = (img_GT-img_GT.min()) / (img_GT.max() - img_GT.min())
                    
            elif self.opt["phase"] == "test":
                img_LR, img_GT = img_LR, img_GT
            elif LR_size is not None:
                if random.random() < 0.5:
                    img_LR, img_GT = self._apply_preprocess(img_LR, img_GT)
            else:
                if random.random() < 0.5:
                    img_LR, img_GT = self._apply_preprocess(img_LR, img_GT)

            return {
                "LQ": _to_tensor(img_LR),
                "GT": _to_tensor(img_GT),
                "LQ_path": LR_path,
                "GT_path": GT_path,
            }


class LQGT_5ch_ref_copied_Dataset_window(LQGT_1ch_Dataset):
    """5-channel LR / GT dataset with sliding-window augmentation.

    Inherits all training logic (crop, augment, optional vignette) from
    :class:`LQGT_1ch_Dataset`.  Overrides image loading to select a
    random window id and read 5 temporal neighbors for that window.
    """

    def _read_window_neighbors(self, lr_path, window_id):
        """Read 5 temporal neighbors for a specific window id."""
        read = lambda p: util.read_img(self.LR_env, p, None)
        suffix = f"_window_{window_id}"
        imgs = []
        for offset in (-2, -1, 0, 1, 2):
            if offset == 0:
                imgs.append(read(lr_path))
            else:
                imgs.append(read(
                    _build_neighbor_path(lr_path, offset, self.LR_env, suffix=suffix)
                ))
        return np.concatenate(imgs, axis=2)

    def _apply_preprocess(self, img_LR, img_GT):
        """Apply Gaussian vignette only when ``opt['vignette']`` is set."""
        if not self.use_vignette:
            return img_LR, img_GT
        
        sx = random.random()*20 + 40
        sy = random.random()*20 + 40
        vignette = _make_gaussian_vignette(self.GT_size, sx=sx, sy=sy)
        return _apply_vignette(img_LR, vignette), _apply_vignette(img_GT, vignette)


    def _read_lr_gt(self, index):
        """Return 5-channel ``(img_LR, img_GT, LR_path, GT_path)``.

        A random ``window_id`` is chosen during training; fixed to 0 at
        test time.
        """
        GT_path = self.GT_paths[index]

        # GT: replicate single channel 5 times
        img_GT_single = util.read_img(self.GT_env, GT_path, None)
        img_GT = np.concatenate([img_GT_single] * 5, axis=2)

        # Select window id
        window_id = random.randint(0, 800)
        if self.opt["phase"] == "test":
            img_LR = np.zeros((9, img_GT.shape[0], img_GT.shape[1], img_GT.shape[2]))
            for window_id, window in zip(range(0, 9), [0, 100, 200, 300, 400, 500, 600, 700, 800]):
                LR_path = f"{self.LR_paths[index][:-4]}_window_{window}.npz"
                print(LR_path)
                img_LR_window = self._read_window_neighbors(LR_path, window_id)
                img_LR[window_id] = img_LR_window

        else:
            LR_path = f"{self.LR_paths[index][:-4]}_window_{window_id}.npz"
            img_LR = self._read_window_neighbors(LR_path, window_id)

        return img_LR, img_GT, LR_path, GT_path

    def __getitem__(self, index):
        GT_size = self.opt["GT_size"]
        LR_size = self.opt["LR_size"]

        img_LR, img_GT, LR_path, GT_path = self._read_lr_gt(index)
       
        if self.opt["phase"] == "test":
            img_LR_windows = np.zeros((img_LR.shape))
            for window_id in range(img_LR.shape[0]):
                img_LR_windows[window_id], img_GT = self._apply_preprocess(img_LR[window_id], img_GT)

            img_LR = img_LR_windows

            return {
                "LQ": torch.from_numpy(
                    np.ascontiguousarray(np.transpose(img_LR, (0, 3, 1, 2)))).float(),
                "GT": _to_tensor(img_GT),
                "LQ_path": LR_path,
                "GT_path": GT_path,
            }

        else:
            if self.opt["phase"] == "train":
                H, W, _ = img_LR.shape
                
                if random.random() < 0.5:
                    img_LR, img_GT = self._apply_preprocess(img_LR, img_GT)

                # randomly crop
                rnd_h = random.randint(0, max(0, H - LR_size))
                rnd_w = random.randint(0, max(0, W - LR_size))
                img_LR = img_LR[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]
                img_GT = img_GT[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]

                img_LR, img_GT = util.augment(
                    [img_LR, img_GT],
                    self.opt["use_flip"], self.opt["use_rot"],
                    self.opt["mode"], self.opt["use_swap"],
                )

                if random.random() < 0.5:
                    img_LR = (img_LR-img_LR.min()) / (img_LR.max() - img_LR.min())
                    img_GT = (img_GT-img_GT.min()) / (img_GT.max() - img_GT.min())

            elif LR_size is not None:
                if random.random() < 0.5:
                    img_LR, img_GT = self._apply_preprocess(img_LR, img_GT)


            else:
                if random.random() < 0.5:
                    img_LR, img_GT = self._apply_preprocess(img_LR, img_GT)

        return {
            "LQ": _to_tensor(img_LR),
            "GT": _to_tensor(img_GT),
            "LQ_path": LR_path,
            "GT_path": GT_path,
        }

    def __len__(self):
        return len(self.GT_paths)



class LQGT_1ch_Dataset_window(LQGT_1ch_Dataset):
    """Single-channel LR / GT image pair dataset.

    Set ``opt["vignette"] = True`` to apply a Gaussian vignette mask.
    Subclass and override :meth:`_read_lr_gt` to change how images are loaded.
    """
    # ------------------------------------------------------------------
    # Override this in subclasses to change the loading behaviour
    # ------------------------------------------------------------------
    def _read_lr_gt(self, index):
        """Return 1-channel ``(img_LR, img_GT, LR_path, GT_path)``.

        A random ``window_id`` is chosen during training; fixed to 0 at
        test time.
        """
        GT_path = self.GT_paths[index]

        # GT: replicate single channel 5 times
        img_GT = util.read_img(self.GT_env, GT_path, None)

        # Select window id
        window_id = random.randint(0, 800)
        if self.opt["phase"] == "test": #TODO
            raise NotImplementedError

        else:
            LR_path = f"{self.LR_paths[index][:-4]}_window_{window_id}.npz"
            img_LR = util.read_img(self.LR_env, LR_path, None)

        return img_LR, img_GT, LR_path, GT_path

    def __getitem__(self, index):
        GT_size = self.opt["GT_size"]
        LR_size = self.opt["LR_size"]

        img_LR, img_GT, LR_path, GT_path = self._read_lr_gt(index)

        if self.opt["phase"] == "train":
            if random.random() < 0.5:
                img_LR, img_GT = self._apply_preprocess(img_LR, img_GT)

            img_LR, img_GT = util.augment(
                [img_LR, img_GT],
                self.opt["use_flip"], self.opt["use_rot"],
                self.opt["mode"], self.opt["use_swap"],
            )

        elif LR_size is not None:
            if random.random() < 0.5:
                img_LR, img_GT = self._apply_preprocess(img_LR, img_GT)

        else:
            if random.random() < 0.5:
                img_LR, img_GT = self._apply_preprocess(img_LR, img_GT)

        return {
            "LQ": _to_tensor(img_LR),
            "GT": _to_tensor(img_GT),
            "LQ_path": LR_path,
            "GT_path": GT_path,
        }


    def __len__(self):
        return len(self.GT_paths)
