
import os
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

def coco_subset_to_binary_masks(subset_dir: str, out_dir: str, mask_value: int = 255):
    """
    Đọc _annotations.coco.json trong subset_dir, tạo mask nhị phân cho từng ảnh:
      - pixel = 0  : background
      - pixel = 255: tumor (hợp nhất mọi annotation thuộc ảnh đó)
    Lưu mask .png vào out_dir với cùng tên file (đổi đuôi .png).
    """
    ann_path = os.path.join(subset_dir, "_annotations.coco.json")
    if not os.path.exists(ann_path):
        print(f"[WARN] Không thấy: {ann_path} -> bỏ qua")
        return
#Tạo file các thư mục đầu ra
    os.makedirs(out_dir, exist_ok=True)
#Truy vấn và đọc file _annotations.coco.json
    coco = COCO(ann_path)
    empty_cnt, total_cnt = 0, 0
    for img_id, img_info in coco.imgs.items():
        total_cnt += 1
        h, w = img_info["height"], img_info["width"]
        img_name = img_info["file_name"]
        #Tách tên file images và tạo tên file masks
        base = os.path.splitext(img_name)[0]
        mask_path = os.path.join(out_dir, f"{base}.png")
#Lấy ID của TẤT CẢ các chú thích (annotations) thuộc về ảnh này.
        ann_ids = coco.getAnnIds(imgIds=[img_id])
#Từ các ID đó, tải toàn bộ thông tin chi tiết của các chú thích.
        anns = coco.loadAnns(ann_ids)
        # Khởi tạo mask background
        mask = np.zeros((h, w), dtype=np.uint8)
        # Hợp nhất tất cả vùng (tumor) -> nhị phân
        # Dùng coco.annToMask để nhận mask nhị phân cho từng ann
        any_fg = False
        for ann in anns:
            m = coco.annToMask(ann)  # {0,1}
            if m is not None:
                mask[m == 1] = 1
                any_fg = any_fg or (m.sum() > 0)

        if not any_fg:
            empty_cnt += 1

        # Lưu ra PNG (0/255 để dễ nhìn; khi training đọc lên rồi >0 -> 1)
        out = (mask * (255 if mask_value == 255 else 1)).astype(np.uint8)
        Image.fromarray(out).save(mask_path)

    print(f"[DONE] {subset_dir} -> {out_dir}")
    print(f"       Tổng ảnh: {total_cnt} | Ảnh không có annotation: {empty_cnt}")

def coco_to_binary_masks(dataset_root="C:/Users/Admin/Desktop/Image Semantic Segmentation Tasks In Medicine/Data/archive", subsets=("train", "valid", "test")):
    for s in subsets:
        subset_dir = os.path.join(dataset_root, s)
        out_dir = os.path.join(dataset_root, "masks", s)
        coco_subset_to_binary_masks(subset_dir, out_dir)

if __name__ == "__main__":
    # Sửa đường dẫn gốc dataset tại đây nếu khác
    coco_to_binary_masks(dataset_root="C:/Users/Admin/Desktop/Image Semantic Segmentation Tasks In Medicine/Data/archive",
                         subsets=("train", "valid", "test"))
