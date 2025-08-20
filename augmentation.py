import os
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
import torch
from torch.utils.data import Dataset, DataLoader

# =========================
# 1. Augmentation pipeline
# =========================
augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05,
                       scale_limit=0.1,
                       rotate_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.2),
    A.ElasticTransform(p=0.2, alpha=1, sigma=50, alpha_affine=50),
])

# =========================
# 2. Augment và lưu từng file .npy
# =========================
def process_split(split_dir, output_dir, n_aug=2):
    img_dir  = os.path.join(split_dir, "images")
    mask_dir = os.path.join(split_dir, "masks")

    img_out = os.path.join(output_dir, "images")
    mask_out = os.path.join(output_dir, "masks")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    img_files = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))]

    idx = 0
    for img_name in tqdm(img_files, desc=f"Processing {split_dir}"):
        img_path = os.path.join(img_dir, img_name)
        mask_path = os.path.join(mask_dir, os.path.splitext(img_name)[0] + ".png")

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"⚠️ Bỏ qua {img_name} vì thiếu ảnh/mask")
            continue

        # ảnh gốc + augmented
        for aug_i in range(n_aug+1):
            if aug_i == 0:
                aug_img, aug_mask = image, mask
            else:
                aug = augmentation(image=image, mask=mask)
                aug_img, aug_mask = aug["image"], aug["mask"]

            # Normalize khi load, ở đây chỉ lưu uint8 để tiết kiệm bộ nhớ
            np.save(os.path.join(img_out, f"img_{idx}.npy"), aug_img.astype(np.uint8))
            np.save(os.path.join(mask_out, f"mask_{idx}.npy"), (aug_mask > 0).astype(np.uint8))
            idx += 1

    print(f"✅ Saved {idx} pairs in {output_dir}")

# =========================
# 3. Dataset class (lazy loading)
# =========================
class BrainMRIDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_files  = sorted(os.listdir(img_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = np.load(os.path.join(self.img_dir, self.img_files[idx]))
        mask = np.load(os.path.join(self.mask_dir, self.mask_files[idx]))

        # Normalize khi load
        img = img.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        img = torch.tensor(img).unsqueeze(0)   # (1,H,W)
        mask = torch.tensor(mask).unsqueeze(0) # (1,H,W)

        return img, mask

# =========================
# 4. Demo
# =========================
if __name__ == "__main__":
    dataset_root = r"C:\Users\Admin\Desktop\Image Semantic Segmentation Tasks In Medicine\Data\archive"
    output_root = os.path.join(dataset_root, "processed")

    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(dataset_root, split)
        output_dir = os.path.join(output_root, split)
        process_split(split_dir, output_dir, n_aug=2)

    # Load thử train
    train_dataset = BrainMRIDataset(
        os.path.join(output_root, "train", "images"),
        os.path.join(output_root, "train", "masks")
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    imgs, masks = next(iter(train_loader))
    print("Batch images:", imgs.shape)  # (B,1,H,W)
    print("Batch masks :", masks.shape) # (B,1,H,W)
