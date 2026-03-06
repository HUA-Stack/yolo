import os
import cv2
from tqdm import tqdm

# 数据集根目录
datasets = [
    r"E:\Vscode_code\yolo\ultralytics-main\datasets\VisDrone2019-VID-train",
    r"E:\Vscode_code\yolo\ultralytics-main\datasets\VisDrone2019-VID-val"
]

# 输出目录映射
output_dirs = {
    "VisDrone2019-VID-train": {
        "images": r"E:\Vscode_code\yolo\ultralytics-main\datasets\VisDrone2019-VID-train\images\train",
        "labels": r"E:\Vscode_code\yolo\ultralytics-main\datasets\VisDrone2019-VID-train\labels\train"
    },
    "VisDrone2019-VID-val": {
        "images": r"E:\Vscode_code\yolo\ultralytics-main\datasets\VisDrone2019-VID-val\images\val",
        "labels": r"E:\Vscode_code\yolo\ultralytics-main\datasets\VisDrone2019-VID-val\labels\val"
    }
}

def convert_bbox(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]

    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh

    return x, y, w, h

for root_dir in datasets:
    dataset_name = os.path.basename(root_dir)
    sequences_dir = os.path.join(root_dir, "sequences")
    annotations_dir = os.path.join(root_dir, "annotations")

    output_images = output_dirs[dataset_name]["images"]
    output_labels = output_dirs[dataset_name]["labels"]

    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_labels, exist_ok=True)

    print(f"开始转换 {dataset_name} → YOLO 格式...")

    for seq in tqdm(os.listdir(sequences_dir)):
        seq_path = os.path.join(sequences_dir, seq)
        anno_path = os.path.join(annotations_dir, seq + ".txt")

        if not os.path.exists(anno_path):
            continue

        # 读取标注文件
        annotations = {}

        with open(anno_path, 'r') as f:
            for line in f.readlines():
                line = line.strip().split(',')
                frame_id = int(line[0])
                x = float(line[2])
                y = float(line[3])
                w = float(line[4])
                h = float(line[5])
                category = int(line[7])

                # 忽略 category = 0 (ignored regions)
                if category == 0:
                    continue

                if frame_id not in annotations:
                    annotations[frame_id] = []

                annotations[frame_id].append([x, y, w, h, category])

        # 遍历图片帧
        images = sorted(os.listdir(seq_path))

        for idx, img_name in enumerate(images):
            frame_id = idx + 1
            img_path = os.path.join(seq_path, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            h_img, w_img = img.shape[:2]

            new_img_name = f"{seq}_{img_name}"
            new_img_path = os.path.join(output_images, new_img_name)
            cv2.imwrite(new_img_path, img)

            label_path = os.path.join(
                output_labels,
                new_img_name.replace(".jpg", ".txt")
            )

            with open(label_path, 'w') as f:
                if frame_id in annotations:
                    for box in annotations[frame_id]:
                        x, y, w, h, category = box
                        x, y, w, h = convert_bbox(
                            (w_img, h_img),
                            (x, y, w, h)
                        )

                        # 类别从1开始 → 改成0开始
                        class_id = category - 1

                        f.write(f"{class_id} {x} {y} {w} {h}\n")

    print(f"{dataset_name} 转换完成！")

print("所有数据集转换完成！")