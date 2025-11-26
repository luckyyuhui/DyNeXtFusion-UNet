import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm


trainval_percent = 1.0
train_percent    = 0.8


VOCdevkit_path = "Dataset_name"

save_base = os.path.join(VOCdevkit_path, "VOC", "ImageSets", "Segmentation")
os.makedirs(save_base, exist_ok=True)

def split_image_sets():
    seg_dir = os.path.join(VOCdevkit_path, "VOC", "SegmentationClass")
    all_files = sorted([f for f in os.listdir(seg_dir) if f.lower().endswith(".png")])
    total = len(all_files)
    indices = list(range(total))

    random.seed(0)
    tv_count = int(total * trainval_percent)
    trainval_idx = random.sample(indices, tv_count)
    train_count = int(tv_count * train_percent)
    train_idx = random.sample(trainval_idx, train_count)

    def write_list(fname, idx_list):
        with open(os.path.join(save_base, fname), "w") as f:
            for i in idx_list:
                name = os.path.splitext(all_files[i])[0]
                f.write(f"{name}\n")

    write_list("trainval.txt", trainval_idx)
    write_list("train.txt", train_idx)
    val_idx = [i for i in trainval_idx if i not in train_idx]
    write_list("val.txt", val_idx)
    test_idx = [i for i in indices if i not in trainval_idx]
    write_list("test.txt", test_idx)

    print("Generate txt in ImageSets.")
    print(f"  total images: {total}")
    print(f"  trainval size: {tv_count}")
    print(f"  train size: {train_count}")
    print("Generate txt in ImageSets done.\n")

def check_dataset_format():
    seg_dir = os.path.join(VOCdevkit_path, "VOC", "SegmentationClass")
    all_files = sorted([f for f in os.listdir(seg_dir) if f.lower().endswith(".png")])

    print("Check datasets format, this may take a while.\n")
    classes_nums = np.zeros(256, dtype=int)

    for fname in tqdm(all_files, desc="Checking"):
        path = os.path.join(seg_dir, fname)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"未检测到标签图片 {path}")

        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.uint8)

        # 维度检查
        if arr.ndim != 2:
            raise ValueError(
                f"标签图片 {fname} 的 shape 为 {arr.shape}，"
                "不属于灰度图或八位彩图，请检查格式。"
            )

        # 累计像素值
        classes_nums += np.bincount(arr.flatten(), minlength=256)

    # 打印统计结果
    print("\nKey  |  Value")
    print("-" * 25)
    for i, cnt in enumerate(classes_nums):
        if cnt > 0:
            print(f"{i:>3d}  |  {cnt}")
    print("-" * 25)

    total_nonzero = classes_nums.sum() - classes_nums[0]
    if classes_nums[255] > 0 and classes_nums[1:255].sum() == 0:
        print("检测到标签值仅包含 0 与 255，二分类场景请将目标值改为 1。")
    elif total_nonzero == 0:
        print("检测到标签中仅包含背景(0)，请检查数据集格式。")
    print("\nFormat check done.")

if __name__ == "__main__":
    split_image_sets()
    check_dataset_format()
