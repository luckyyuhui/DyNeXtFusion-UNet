import os
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def _read_image_as_array(path):
    with Image.open(path) as img:
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        if img.mode == 'L':
            arr = np.asarray(img, dtype=np.float64)
            arr = arr[:, :, None]
        else:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            arr = np.asarray(img, dtype=np.float64)
        max_val = arr.max()
        if max_val > 1.0:
            if max_val <= 255:
                arr = arr / 255.0
            elif max_val <= 65535:
                arr = arr / 65535.0
            else:
                arr = arr / max_val
        return arr

def compute_mean_std(image_dir, image_list_txt):
    assert os.path.isdir(image_dir),    f"image_dir does not exist: {image_dir}"
    assert os.path.isfile(image_list_txt), f"image_list_txt does not exist: {image_list_txt}"

    with open(image_list_txt, 'r') as f:
        img_ids = [ln.strip() for ln in f if ln.strip()]

    pixel_sum      = None
    pixel_sq_sum   = None
    pixel_count    = 0
    used_images    = 0

    for img_id in img_ids:
        candidates = []
        if os.path.splitext(img_id)[1]:
            candidates.append(os.path.join(image_dir, img_id))
        else:
            candidates.append(os.path.join(image_dir, img_id + ".jpg"))
            candidates.append(os.path.join(image_dir, img_id + ".jpeg"))
            candidates.append(os.path.join(image_dir, img_id + ".png"))
            candidates.append(os.path.join(image_dir, img_id + ".tif"))
            candidates.append(os.path.join(image_dir, img_id + ".tiff"))

        path = None
        for c in candidates:
            if os.path.isfile(c):
                path = c
                break

        if path is None:
            print(f"[Skip] Not found: {os.path.join(image_dir, img_id)}")
            continue

        try:
            arr = _read_image_as_array(path)  # H, W, C 或 H, W
        except Exception as e:
            print(f"[Skip] Not found {path}: {e}")
            continue

        if arr.ndim == 3 and arr.shape[0] <= 10 and arr.shape[0] < arr.shape[2]:
            arr = np.transpose(arr, (1, 2, 0))

        if arr.ndim == 2:
            arr = arr[:, :, None]

        h, w, c = arr.shape
        flat    = arr.reshape(-1, c)

        if pixel_sum is None:
            pixel_sum    = np.zeros(c, dtype=np.float64)
            pixel_sq_sum = np.zeros(c, dtype=np.float64)

        pixel_sum    += flat.sum(axis=0)
        pixel_sq_sum += (flat ** 2).sum(axis=0)
        pixel_count  += h * w
        used_images  += 1

    if used_images == 0:
        raise RuntimeError("No image was successfully read. Check the path and the list file.")

    # 最终计算
    mean = pixel_sum / pixel_count
    std  = np.sqrt(pixel_sq_sum / pixel_count - mean ** 2)

    print("==== Number of image channels:", len(mean), "====")
    for i, (m, s) in enumerate(zip(mean, std)):
        print(f"  channel {i}: mean = {m:.6f}, std = {s:.6f}")

    return mean, std


if __name__ == "__main__":
    image_dir      = "./Dataset_name/VOC/JPEGImages"
    image_list_txt = "./Dataset_name/VOC/ImageSets/Segmentation/val.txt"
    compute_mean_std(image_dir, image_list_txt)
