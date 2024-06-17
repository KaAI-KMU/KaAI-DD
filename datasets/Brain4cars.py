import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
from os.path import *
import numpy as np
import random
from glob import glob
import csv
from utils import load_value_file

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pandas as pd


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader_inside(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, '{:06d}.png'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            print(image_path)
            return video

    return video

def get_default_video_loader_inside():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader_inside, image_loader=image_loader)

def video_loader_outside(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, '{:06d}.png'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            print(image_path)
            return video

    return video

def get_default_video_loader_outside():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader_outside, image_loader=image_loader)


def test_load_annotation_data(data_file_path, fold):
    database = {}
    data_file_path = os.path.join(data_file_path, 'test.csv')
    print('Load from %s'%data_file_path)
    with open(data_file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            value = {}
            value['subset'] = row[3]
            value['label'] = row[1]
            value['n_frames'] = int(row[2])
            database[row[0]] = valuetraining_data_inside
    return database

def load_annotation_data(data_file_path, fold):
    database = {}
    data_file_path = os.path.join(data_file_path, 'fold%d.csv'%fold)
    print('Load from %s'%data_file_path)
    with open(data_file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            value = {}
            value['subset'] = row[3]
            value['label'] = row[1]
            value['n_frames'] = int(row[2])
            database[row[0]] = value
    return database

def get_class_labels():
#### define the labels map
    class_labels_map = {}
    class_labels_map['end_action'] = 0
    class_labels_map['lchange'] = 1
    class_labels_map['lturn'] = 2
    class_labels_map['rchange'] = 3
    class_labels_map['rturn'] = 4
    return class_labels_map

def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data.items():
        this_subset = value['subset']
        if this_subset == subset:
            label = value['label']
            video_names.append(key)    ### key = 'rturn/20141220_154451_747_897'
            annotations.append(value)

    return video_names, annotations


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video, end_second,
                 sample_duration, fold):
    
    if subset != "test":
        data = load_annotation_data(annotation_path, fold)
    else:
        data = test_load_annotation_data(annotation_path, fold)

    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels()
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    for i in range(len(video_names)):
        if i % 100 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i])
        if not os.path.exists(video_path):
            print('File does not exists: %s'%video_path)
            continue

#        n_frames = annotations[i]['n_frames']
        # count in the dir
        l = os.listdir(video_path)
        # If there are other files (e.g. original videos) besides the images in the folder, please abstract.
        n_frames = len(l)-1

        # if n_frames < 16 + 25*(end_second-1):
        #     print('Video is too short: %s'%video_path)
        #     continue

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,

            'video_id': video_names[i].split('/')[1]
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(1, n_frames + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                for j in range(0, n_samples_for_each_video):
                    sample['frame_indices'] = list(range(1, n_frames+1))
                    sample_j = copy.deepcopy(sample)
                    dataset.append(sample_j)

    return dataset, idx_to_class

def make_dataset_outside(root_path, annotation_path, subset, n_samples_for_each_video, end_second,
                 sample_duration, fold):
    if subset != "test":
        data = load_annotation_data(annotation_path, fold)
    else:
        data = test_load_annotation_data(annotation_path, fold)

    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels()
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    for i in range(len(video_names)):
        if i % 100 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i])
        if not os.path.exists(video_path):
            print('File does not exists: %s'%video_path)
            continue

#        n_frames = annotations[i]['n_frames']
        # count in the dir
        l = os.listdir(video_path+'flir4')
        # If there are other files (e.g. original videos) besides the images in the folder, please abstract.
        n_frames = len(l)-1

        # if n_frames < 16 + 30*(end_second-1):
        #     print('Video is too short: %s'%video_path)
        #     continue

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,

            'video_id': video_names[i].split('/')[1]
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(0, n_frames + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                for j in range(0, n_samples_for_each_video):
                    sample['frame_indices'] = list(range(0, n_frames+1))
                    sample_j = copy.deepcopy(sample)
                    dataset.append(sample_j)
    return dataset, idx_to_class


class Brain4cars_Inside(data.Dataset):
    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 nfold, 
                 end_second,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 horizontal_flip=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=5,
                 get_loader=get_default_video_loader_inside):
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            end_second, sample_duration, nfold)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.horizontal_flip = horizontal_flip
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']
        h_flip = False

        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        
        if self.horizontal_flip is not None:
            p = random.random()
            if p < 0.5:
                h_flip = True
                clip = [self.horizontal_flip(img) for img in clip]

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        if (h_flip == True) and (target != 0):
            if target == 1:
                target = 3
            elif target == 3:
                target = 1
            elif target == 2:
                target = 4
            elif target == 4:
                target = 2

        return clip, target
    
    def __len__(self):
        return len(self.data)

class Brain4cars_Outside(data.Dataset):
    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 nfold,
                 end_second,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 horizontal_flip=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader_outside):
        self.data, self.class_names = make_dataset_outside(
            root_path, annotation_path, subset, n_samples_for_each_video,
            end_second, sample_duration, nfold)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.horizontal_flip = horizontal_flip
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is an image.
        """
        # 원본
        # path = self.data[index]['video']

        # frame_indices = self.data[index]['frame_indices']


        # h_flip = False

        # if self.temporal_transform is not None:
        #     frame_indices,target_idc = self.temporal_transform(frame_indices)
        # clip = self.loader(path, frame_indices)
        # target = self.loader(path, target_idc)

        # if self.horizontal_flip is not None:
        #     p = random.random()
        #     if p < 0.5:
        #         clip = [self.horizontal_flip(img) for img in clip]
        #         target = [self.horizontal_flip(img) for img in target]

        # if self.spatial_transform is not None:
        #     self.spatial_transform.randomize_parameters()
        #     clip = [self.spatial_transform(img) for img in clip]
        # clip = torch.stack(clip, 0)

        # if self.target_transform is not None:
        #     target = [self.target_transform(img) for img in target]
        # target = torch.stack(target, 0).permute(1, 0, 2, 3).squeeze()
            

        # return clip, target

        #4방향
        
        video_data = self.data[index]
        path = video_data['video']
        frame_indices = video_data['frame_indices']

        if self.temporal_transform is not None:
            frame_indices, target_idc = self.temporal_transform(frame_indices)

        clips = {}
        targets = {}
        # directions = ['flir4', 'flir1', 'flir2', 'flir3']
        directions = ['flir4']

        for direction in directions:
            video_dir_path = os.path.join(path, direction)
            video_frames = self.loader(video_dir_path, frame_indices)
            target_frames = self.loader(video_dir_path, target_idc)

            # Apply horizontal flip and target change only for flir4 and flir3
            if self.horizontal_flip is not None:
                p = random.random()
                if p < 0.5:
                    video_frames = [self.horizontal_flip(img) for img in video_frames]
                    target_frames = [self.horizontal_flip(img) for img in target_frames]

            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                video_frames = [self.spatial_transform(img) for img in video_frames]

            video_frames = torch.stack(video_frames, 0)
            clips[direction] = video_frames

            if self.target_transform is not None:
                target_frames = [self.target_transform(img) for img in target_frames]
            target_frames = torch.stack(target_frames, 0).permute(1, 0, 2, 3).squeeze()
            targets[direction] = target_frames

            # 중간 데이터 삭제
            del video_frames
            del target_frames
            torch.cuda.empty_cache()  # GPU 캐시 메모리 정리

        return clips, targets



    def __len__(self):
        return len(self.data)
    

# def make_dataset_gaze(annotation_path, subset, fold):
#     """
#     Gaze 데이터를 로딩하기 위한 함수.
    
#     Args:
#         annotation_path (str): Annotation 파일의 경로.
#         subset (str): 데이터셋의 부분 집합 ('training', 'validation', 'test').
#         fold (int): 사용할 fold 번호.
    
#     Returns:
#         list: 데이터셋을 구성하는 샘플들의 리스트.
#     """
#     # Annotation 데이터 로딩
#     if subset != "test":
#         data = load_annotation_data(annotation_path, fold)
#     else:
#         data = test_load_annotation_data(annotation_path, fold)

#     video_names, annotations = get_video_names_and_annotations(data, subset)

#     class_to_idx = get_class_labels()
#     idx_to_class = {}
#     for name, label in class_to_idx.items():
#         idx_to_class[label] = name

#     dataset = []
#     for i, video_name in enumerate(video_names):
#         gaze_data_path = os.path.join(annotation_path,'Gaze&CAN',video_name,"Gaze_point.csv")  # 가정: Gaze 데이터 파일명
#         if not os.path.exists(gaze_data_path):
#             print(f"Gaze data file does not exist: {gaze_data_path}")
#             continue

#         # Gaze 데이터 로딩
#         gaze_data = pd.read_csv(gaze_data_path)
#         # Gaze 데이터 형식에 따라 필요한 처리 수행

#         # 데이터셋 샘플 생성
#         sample = {
#             'video_id': video_name,
#             'gaze_data': gaze_data,  # 여기서는 pandas DataFrame을 직접 사용하고 있으나, 필요에 따라 numpy 배열 등으로 변환 가능
#             'label': class_to_idx[annotations[i]['label']] if len(annotations) != 0 else -1
#         }

#         dataset.append(sample)
    
#     return dataset, idx_to_class
    


def make_dataset_gaze(annotation_path, subset, fold, n_samples_for_each_video=10):
    if subset != "test":
        data = load_annotation_data(annotation_path, fold)
    else:
        data = test_load_annotation_data(annotation_path, fold)

    video_names, annotations = get_video_names_and_annotations(data, subset)

    class_to_idx = get_class_labels()
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    for i, video_name in enumerate(video_names):
        gaze_data_path = os.path.join(annotation_path,'Gaze&CAN',video_name,"Gaze_point.csv")
        if not os.path.exists(gaze_data_path):
            print(f"Gaze data file does not exist: {gaze_data_path}")
            continue

        gaze_data = pd.read_csv(gaze_data_path)

        sample = {
            'video_id': video_name,
            'gaze_data': gaze_data,
            'label': class_to_idx[annotations[i]['label']] if len(annotations) != 0 else -1
        }
        dataset.append(sample)
    
    return dataset, idx_to_class



class GazeDataset(data.Dataset):
    def __init__(self, annotation_path, subset, fold):
        """
        Gaze 데이터셋을 위한 Dataset 클래스 초기화.
        
        Args:
            annotation_path (str): Annotation 파일의 경로.
            subset (str): 데이터셋의 부분 집합 ('training', 'validation', 'test').
            fold (int): 사용할 fold 번호.
        """
        self.dataset, self.idx_to_class = make_dataset_gaze(annotation_path, subset, fold)
        
    def __len__(self):
        """데이터셋의 샘플 개수를 반환합니다."""
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        주어진 인덱스에 해당하는 샘플을 데이터셋에서 가져옵니다.
        Args:
            idx (int): 가져올 샘플의 인덱스.
        Returns:
            tuple: (gaze_data, label) - gaze 데이터와 레이블을 포함하는 튜플.
        """
        sample = self.dataset[idx]
        gaze_data = sample['gaze_data'].values  # DataFrame을 numpy 배열로 변환
        label = sample['label']
        
        # 뒤에서부터 20개의 프레임만 선택
        if len(gaze_data) > 20:
            gaze_data = gaze_data[-20:]
        # 데이터가 20개 미만인 경우, 전체 데이터 사용 (또는 다른 로직 적용 가능)

        # 데이터 타입 변환
        gaze_data = torch.tensor(gaze_data, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)
        
        return gaze_data, label