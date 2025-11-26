import os
import glob
import numpy as np
import tifffile
from PIL import Image
import torch
from torch.utils.data import Dataset

class VOCSegmentation(Dataset):
    def __init__(
        self,
        voc_root,
        transforms=None,
        txt_name="train.txt",
        num_classes=2,
        ignore_index=255
    ):
        super().__init__()
        root = os.path.join(voc_root, "Dataset_name", "VOC")
        assert os.path.isdir(root), f"未找到 VOC 根目录: {root}"

        self.image_dir    = os.path.join(root, "JPEGImages")
        self.mask_dir     = os.path.join(root, "SegmentationClass")
        list_dir          = os.path.join(root, "ImageSets", "Segmentation")
        txt_path          = os.path.join(list_dir, txt_name)
        assert os.path.isdir(self.image_dir), f"影像目录不存在: {self.image_dir}"
        assert os.path.isdir(self.mask_dir),  f"标签目录不存在: {self.mask_dir}"
        assert os.path.isfile(txt_path),      f"列表文件不存在: {txt_path}"

        with open(txt_path, "r") as f:
            names = [ln.strip() for ln in f if ln.strip()]
        # 保持原有命名约定（image: .jpg, mask: .png）
        self.images = [os.path.join(self.image_dir, nm + ".jpg") for nm in names]
        self.masks  = [os.path.join(self.mask_dir,  nm + ".png") for nm in names]
        assert len(self.images) == len(self.masks), "影像与标签数量不一致"

        self.transforms   = transforms
        self.num_classes  = num_classes
        self.ignore_index = ignore_index

    def __len__(self):
        return len(self.images)

    def _read_image(self, img_path):
        """
        兼容读取多种格式：
        - 对于 tif/tiff 使用 tifffile.imread
        - 对于 jpg/png 等使用 PIL.Image.open
        返回 numpy array，shape=(H,W,C) 或 (H,W,1)（uint8/uint16 视原图而定）
        """
        suffix = str(img_path).lower().split('.')[-1]
        if suffix in ("tif", "tiff"):
            arr = tifffile.imread(img_path)
        else:
            with Image.open(img_path) as im:
                # 保持原有通道数（若是灰度返回 L -> single channel）
                # 将 PIL image 转为 numpy array
                arr = np.array(im)
        # 统一处理：若是 (C,H,W) 且 C 小于等于 10 且 C < W，说明通道维在前，将其转到最后
        if arr.ndim == 3 and arr.shape[0] <= 10 and arr.shape[0] < arr.shape[2]:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 2:
            arr = arr[:, :, None]
        return arr

    def __getitem__(self, idx):
        img_path = self.images[idx]
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        arr = self._read_image(img_path)
        h, w, c = arr.shape

        if c == 1:
            arr3 = np.tile(arr, (1, 1, 3))
            img = Image.fromarray(arr3.astype(np.uint8), mode="RGB")
        elif c == 3:
            img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
        elif c == 4:
            img = Image.fromarray(arr.astype(np.uint8), mode="RGBA")
        else:
            raise ValueError(f"不支持的通道数: {c} for {img_path}")

        mask_path = self.masks[idx]
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        mask = Image.open(mask_path)
        if mask.mode != "L":
            mask = mask.convert("L")

        if self.transforms is not None:
            result = self.transforms(img, mask)
            if result is None:
                raise RuntimeError(
                    f"[Dataset] transforms 返回 None at idx={idx}"
                )
            img, mask = result

        if img is None or mask is None:
            raise RuntimeError(
                f"[Dataset] img or mask is None after transforms at idx={idx}"
            )

        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(np.array(mask)).long()
        else:
            mask = mask.long()

        mask = torch.clamp(mask, 0, self.num_classes - 1)
        mask[mask >= self.num_classes] = self.ignore_index

        return img, mask

    @staticmethod
    def collate_fn(batch):
        """
        batch: list of (img_tensor[C,H,W], mask_tensor[H,W] or [1,H,W])
        返回:
            images: [B,C,H_max,W_max]
            masks:  [B,H,W]
        """
        valid_batch = [(i, m) for i, m in batch if i is not None and m is not None]
        if len(valid_batch) < len(batch):
            print(f"[Collate] drop {len(batch) - len(valid_batch)} invalid samples")
        if not valid_batch:
            raise RuntimeError("[Collate] 全部样本无效，batch 为空！")

        imgs, masks = zip(*valid_batch)
        batched_imgs = _cat_list(imgs, fill_value=0)

        fixed_masks = []
        for m in masks:
            if m.ndim == 3 and m.shape[0] == 1:
                m = m.squeeze(0)
            fixed_masks.append(m)
        batched_masks = torch.stack(fixed_masks, dim=0)

        return batched_imgs, batched_masks

def _cat_list(tensors, fill_value=0):
    """
    把 list of [C,H,W] 拼成 batch [B,C,H_max,W_max]，空白处填 fill_value
    """
    for t in tensors:
        if t.ndim != 3:
            raise ValueError(f"Expect 3D tensor, got {t.shape}")
    max_h = max(t.shape[1] for t in tensors)
    max_w = max(t.shape[2] for t in tensors)
    c     = tensors[0].shape[0]
    batch_shape = (len(tensors), c, max_h, max_w)

    batched = tensors[0].new_full(batch_shape, fill_value)
    for t, pad in zip(tensors, batched):
        _, h, w = t.shape
        pad[:, :h, :w].copy_(t)
    return batched
