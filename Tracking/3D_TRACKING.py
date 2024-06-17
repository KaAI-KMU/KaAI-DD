import json
import os
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import open3d as o3d

# 칼만 필터를 생성 및 초기화하는 함수
def create_kalman_filter():
    kf = KalmanFilter(dim_x=7, dim_z=3)
    kf.F = np.array([
        [1, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1]
    ])
    kf.H = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0]
    ])
    kf.P[3:, 3:] *= 1000.  # 초기 속도에 대한 높은 불확실성
    kf.P *= 10.  # 초기 불확실성
    kf.R[2, 2] *= 0.01  # 측정 노이즈
    return kf

# 사전 학습된 ResNet50 모델을 사용하여 이미지에서 특징을 추출하는 클래스
class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.transforms = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract(self, image):
        image = self.transforms(image).unsqueeze(0)
        with torch.no_grad():
            feature = self.model(image)
        return feature.squeeze(0).cpu().numpy()

# 데이터셋, 이미지, 라이다 파일의 경로 설정
root_path = "/media/seongyong/T7"
dataset_base_path = os.path.join(root_path, "3d_label_file/  (건대)")
image_base_path = os.path.join(root_path, "kaai_dataset/KonkukUniv(건대)")
lidar_base_path = os.path.join(root_path, "kaai_dataset/KonkukUniv(건대)")

# 사전 학습된 ResNet50 모델을 로드하고 마지막 완전 연결 층을 제거
model = resnet50(pretrained=True)
model.fc = torch.nn.Identity()
feature_extractor = FeatureExtractor(model)

# 데이터셋이 포함된 각 폴더를 처리하는 함수
def process_folder(dataset_path, image_path, lidar_path):
    files = sorted([f for f in os.listdir(dataset_path) if f.endswith('.json')])
    track_id_counter = 0
    tracks = {}

    # 다양한 매트릭스에 대한 가중치 설정
    weight_position = 0.2
    weight_feature = 0.5
    weight_iou = 0.3
    time_delta = 1
    initial_velocity_threshold = 100.0
    velocity_threshold_increment = 50.0

    # 두 3D 경계 상자 간의 IoU(교차합)를 계산하는 함수
    def calculate_iou(box1, box2):
        def volume(b):
            return (b['scale']['x']) * (b['scale']['y']) * (b['scale']['z'])

        def intersect(b1, b2):
            x_min = max(b1['position']['x'] - b1['scale']['x'] / 2, b2['position']['x'] - b2['scale']['x'] / 2)
            x_max = min(b1['position']['x'] + b1['scale']['x'] / 2, b2['position']['x'] + b2['scale']['x'] / 2)
            y_min = max(b1['position']['y'] - b1['scale']['y'] / 2, b2['position']['y'] - b2['scale']['y'] / 2)
            y_max = min(b1['position']['y'] + b1['scale']['y'] / 2, b2['position']['y'] + b2['scale']['y'] / 2)
            z_min = max(b1['position']['z'] - b1['scale']['z'] / 2, b2['position']['z'] - b2['scale']['z'] / 2)
            z_max = min(b1['position']['z'] + b1['scale']['z'] / 2, b2['position']['z'] + b2['scale']['z'] / 2)

            if x_min < x_max and y_min < y_max and z_min < z_max:
                return (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
            else:
                return 0.0

        intersection = intersect(box1, box2)
        union = volume(box1) + volume(box2) - intersection
        return intersection / union

    # 두 위치 간의 상대 속도를 계산하는 함수
    def calculate_relative_velocity(pos1, pos2, time_delta):
        dx = (pos2[0] - pos1[0]) / time_delta
        dy = (pos2[1] - pos1[1]) / time_delta
        dz = (pos2[2] - pos1[2]) / time_delta
        return np.sqrt(dx**2 + dy**2 + dz**2)

    # 데이터셋 파일을 순회하면서 처리
    for i in range(len(files) - 1):
        try:
            with open(os.path.join(dataset_path, files[i])) as f1, open(os.path.join(dataset_path, files[i + 1])) as f2:
                data1 = json.load(f1)
                data2 = json.load(f2)

                # 첫 번째 프레임으로 트랙 초기화
                if i == 0:
                    for obj in data1:
                        kf = create_kalman_filter()
                        kf.x[:3] = np.array([obj['psr']['position']['x'], obj['psr']['position']['y'], obj['psr']['position']['z']]).reshape((3, 1))
                        kf.x[3:6] = 0
                        tracks[track_id_counter] = {'kf': kf, 'obj': obj, 'feature': None}
                        obj['obj_id'] = str(track_id_counter)
                        track_id_counter += 1

                    output_file = os.path.join(dataset_path, files[i])
                    with open(output_file, 'w') as f:
                        json.dump(data1, f, indent=4)

                detections = []
                frame_number = int(files[i + 1].split('.')[0])
                image_file = os.path.join(image_path, f"{frame_number:06d}.png")
                lidar_file = os.path.join(lidar_path, f"{frame_number:06d}.pcd")

                image = Image.open(image_file).convert('RGB')
                feature = feature_extractor.extract(image)

                pcd = o3d.io.read_point_cloud(lidar_file)
                lidar_points = np.asarray(pcd.points)

                # 다음 프레임에서 검출된 객체 수집
                for obj in data2:
                    detections.append({'obj': obj, 'feature': feature})

                track_ids = list(tracks.keys())
                track_positions = [track['kf'].x[:3].flatten() for track in tracks.values()]
                track_features = [track['feature'] for track in tracks.values() if track['feature'] is not None]
                detection_positions = np.array([[d['obj']['psr']['position']['x'], d['obj']['psr']['position']['y'], d['obj']['psr']['position']['z']] for d in detections])
                detection_features = [d['feature'] for d in detections]

                if len(track_features) > 0:
                    track_features = np.array(track_features)
                else:
                    track_features = np.empty((0, feature.shape[0]))

                if len(detection_features) > 0:
                    detection_features = np.array(detection_features)
                else:
                    detection_features = np.empty((0, feature.shape[0]))

                # 매칭을 위한 비용 행렬 계산
                if len(track_positions) > 0 and len(detection_positions) > 0:
                    cost_matrix = cdist(detection_positions, track_positions, metric='euclidean')
                    if track_features.shape[0] > 0 and detection_features.shape[0] > 0:
                        feature_cost_matrix = cdist(detection_features, track_features, metric='cosine')
                        cost

_matrix = weight_position * cost_matrix + weight_feature * feature_cost_matrix

                    iou_cost_matrix = np.zeros((len(detections), len(tracks)))
                    for d_idx, detection in enumerate(detections):
                        for t_idx, track_id in enumerate(track_ids):
                            iou_cost_matrix[d_idx, t_idx] = calculate_iou(detection['obj']['psr'], tracks[track_id]['obj']['psr'])
                    cost_matrix = weight_position * cost_matrix + weight_iou * (1 - iou_cost_matrix)
                else:
                    cost_matrix = np.empty((len(detection_positions), len(track_positions)))

                # 할당 문제 해결
                matched_indices = linear_sum_assignment(cost_matrix)
                unmatched_detections = set(range(len(detections))) - set(matched_indices[0])
                unmatched_tracks = set(range(len(tracks))) - set(matched_indices[1])

                velocity_threshold = initial_velocity_threshold + (i * velocity_threshold_increment)

                # 매칭된 검출로 트랙 업데이트
                for d, t in zip(*matched_indices):
                    if cost_matrix[d, t] < 1.0:
                        detection_velocity = (detection_positions[d] - track_positions[t]) / time_delta
                        relative_velocity = calculate_relative_velocity(track_positions[t], detection_positions[d], time_delta)
                        if relative_velocity < velocity_threshold:
                            tracks[track_ids[t]]['kf'].update(detection_positions[d].reshape((3, 1)))
                            tracks[track_ids[t]]['kf'].x[3:6] = detection_velocity.reshape((3, 1))
                            tracks[track_ids[t]]['obj'] = detections[d]['obj']
                            tracks[track_ids[t]]['feature'] = detections[d]['feature']
                            data2[d]['obj_id'] = str(track_ids[t])
                        else:
                            unmatched_detections.add(d)
                            unmatched_tracks.add(t)
                    else:
                        unmatched_detections.add(d)
                        unmatched_tracks.add(t)

                # 매칭되지 않은 검출에 대해 새로운 트랙 생성
                for d in unmatched_detections:
                    kf = create_kalman_filter()
                    kf.x[:3] = detection_positions[d].reshape((3, 1))
                    kf.x[3:6] = 0
                    tracks[track_id_counter] = {'kf': kf, 'obj': detections[d]['obj'], 'feature': detections[d]['feature']}
                    data2[d]['obj_id'] = str(track_id_counter)
                    track_id_counter += 1

                # 매칭되지 않은 트랙 제거
                for t in unmatched_tracks:
                    tracks.pop(track_ids[t])

                # 모든 트랙에 대해 다음 위치 예측
                for track in tracks.values():
                    track['kf'].predict()

                # 업데이트된 트랙을 출력 파일에 저장
                output_file = os.path.join(dataset_path, files[i + 1])
                with open(output_file, 'w') as f:
                    json.dump(data2, f, indent=4)
        except Exception as e:
            print(f"Error processing frame {i} in folder {dataset_path}: {e}")

# 처리할 모든 폴더를 순회하며 처리
folder_ids = [1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]  # 처리할 폴더 ID 리스트
for folder_id in folder_ids:
    dataset_path = os.path.join(dataset_base_path, str(folder_id))
    image_path = os.path.join(image_base_path, f"20240118_drive_{folder_id:03d}/flir4")
    lidar_path = os.path.join(lidar_base_path, f"20240118_drive_{folder_id:03d}/lidar")

    if os.path.exists(dataset_path) and os.path.exists(image_path) and os.path.exists(lidar_path):
        try:
            process_folder(dataset_path, image_path, lidar_path)
        except Exception as e:
            print(f"Error processing folder {folder_id}: {e}")

print("모든 트랙킹 작업 완료")  # 모든 트랙킹 작업 완료
