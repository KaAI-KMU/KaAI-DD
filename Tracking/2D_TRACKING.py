import os
import glob

# IoU 계산 함수
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxB[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# 텍스트 파일에서 바운딩 박스 정보 읽기 함수
def read_boxes(file_path):
    boxes = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                print(f"Skipping malformed line in {file_path}: {line.strip()}")
                continue
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:5])
            xmin = x_center - (width / 2)
            ymin = y_center - (height / 2)
            xmax = x_center + (width / 2)
            ymax = y_center + (height / 2)
            boxes.append((xmin, ymin, xmax, ymax, class_id))
    return boxes

# 트랙킹 ID 할당 및 텍스트 파일 덮어쓰기 함수
def assign_tracking_ids(annotations_path):
    try:
        files = sorted(glob.glob(os.path.join(annotations_path, '*.txt')))
        if not files:
            print(f"No annotation files found in {annotations_path}. Skipping...")
            return
        
        tracking_id = 0
        last_frame_boxes = []
        for file_path in files:
            current_frame_boxes = read_boxes(file_path)
            used_ids = set()
            if not last_frame_boxes:
                for i in range(len(current_frame_boxes)):
                    current_frame_boxes[i] += (tracking_id,)
                    used_ids.add(tracking_id)
                    tracking_id += 1
            else:
                for i, box in enumerate(current_frame_boxes):
                    max_iou = 0
                    max_iou_id = -1
                    for last_box in last_frame_boxes:
                        iou = calculate_iou(box[:4], last_box[:4])
                        if iou > max_iou and last_box[5] not in used_ids:
                            max_iou = iou
                            max_iou_id = last_box[5]
                    if max_iou_id != -1:
                        current_frame_boxes[i] += (max_iou_id,)
                        used_ids.add(max_iou_id)
                    else:
                        current_frame_boxes[i] += (tracking_id,)
                        used_ids.add(tracking_id)
                        tracking_id += 1
            last_frame_boxes = current_frame_boxes
            with open(file_path, 'w') as f:
                for box in current_frame_boxes:
                    xmin, ymin, xmax, ymax, class_id, tid = box
                    x_center = (xmin + xmax) / 2
                    y_center = (ymin + ymax) / 2
                    width = xmax - xmin
                    height = ymax - ymin
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {tid}\n")
    except Exception as e:
        print(f"An error occurred while processing {annotations_path}: {e}")

# 모든 폴더에 대해 트랙킹 ID 할당
base_path = "/media/seongyong/T7/2d_label_file/Yongsan(용산)"
main_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f)) and f.isdigit()]
main_folders = sorted(main_folders, key=int)

for folder in main_folders:
    folder_path = os.path.join(base_path, folder)
    subfolders = [f for f in os.listdir(folder_path) if f.startswith("flir4")]
    for subfolder in subfolders:
        annotations_path = os.path.join(folder_path, subfolder)
        assign_tracking_ids(annotations_path)
