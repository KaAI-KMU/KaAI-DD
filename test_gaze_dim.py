import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from tqdm import tqdm
from opts import parse_opts
from dataset import get_training_set_gaze, get_validation_set_gaze


class GazePredictorLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, drop_rate):
        super(GazePredictorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

def train_epoch(model, criterion, optimizer, data_loader, device):
    model.train()
    running_loss = 0.0
    total_samples = 0
    correct_predictions = 0

    for inputs, labels in tqdm(data_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples

    return epoch_loss, epoch_acc

def validate_epoch(model, criterion, data_loader, device):
    model.eval()
    running_loss = 0.0
    total_samples = 0
    correct_predictions = 0

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples

    return epoch_loss, epoch_acc

def find_best_hidden_dim(train_loader, val_loader, input_dim, num_layers, num_classes, device):
    hidden_dims = [256, 512, 1024, 2048]  # Example hidden dimensions to try
    best_hidden_dim = None
    best_validation_loss = float('inf')

    for hidden_dim in hidden_dims:
        print(f"Testing with hidden_dim = {hidden_dim}")
        model = GazePredictorLSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, num_classes=num_classes, drop_rate=0.5).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Training
        train_loss, train_acc = train_epoch(model, criterion, optimizer, train_loader, device)
        print(f"Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        # Validation
        val_loss, val_acc = validate_epoch(model, criterion, val_loader, device)
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            best_hidden_dim = hidden_dim

    return best_hidden_dim

opt = parse_opts()

# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_data_gaze = get_training_set_gaze(opt)
train_loader_gaze = torch.utils.data.DataLoader(
    training_data_gaze,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_threads,
    pin_memory=True,
    drop_last=True
)


val_data_gaze = get_validation_set_gaze(opt)
val_loader_gaze = torch.utils.data.DataLoader(
    val_data_gaze,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_threads,
    pin_memory=True,
    drop_last=True
)

# Load your train_loader and val_loader here
# train_loader = DataLoader(...)
# val_loader = DataLoader(...)

# Assuming GazePredictorLSTM class is defined elsewhere in your code
for i in range(40):
    best_hidden_dim = find_best_hidden_dim(train_loader_gaze, val_loader_gaze, input_dim=2, num_layers=2, num_classes=5, device=device)
    print(f"Best hidden_dim found: {best_hidden_dim}")


# import os
# import sys
# import json
# import torch
# import torch.nn as nn
# # from main_inside import train_inside
# from opts import parse_opts
# from model import generate_model
# from dataset import get_training_set_inside,get_training_set_outside, get_validation_set_inside, get_validation_set_outside, get_training_set_gaze, get_validation_set_gaze
# from spatial_transforms import (
#     Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
#     MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor, DriverFocusCrop, DriverCenterCrop)
# from temporal_transforms import LoopPadding, TemporalRandomCrop, TemporalCenterCrop, UniformRandomSample, UniformEndSample, UniformIntervalCrop

# from target_transforms import ClassLabel, VideoID
# from target_transforms import Compose as TargetCompose
# from torch.autograd import Variable
# from mean import get_mean, get_std
# from models.convolution_lstm import encoder,classifier
# from utils import AverageMeter, calculate_accuracy
# import warnings
# from tqdm import tqdm
# #import matplotlib.pyplot as plt
# from torch import optim
# from torch.optim import lr_scheduler
# import time
# from utils import Logger
# from torch.utils.tensorboard import SummaryWriter
# import torch.nn.functional as F



# # 경고 무시 설정
# warnings.filterwarnings("ignore", category=UserWarning)

# # 내부, 외부에서 추출한 벡터를 합쳐서 분류기에 넣어줌
# class conv_classifier(nn.Module):
#     def __init__(self):
#         super(conv_classifier, self).__init__()
        
#         # max pool함
#         self.conv_block = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0),
#             nn.ReLU(),           
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
#             nn.BatchNorm2d(64),
            
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
#             nn.BatchNorm2d(128),           
            
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),   
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
#             nn.BatchNorm2d(256),
            
#             nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
#             nn.BatchNorm2d(512),
            
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=3, padding=0),
#             nn.BatchNorm2d(512)     
                  
#         )

#         self.classifier_fc = nn.Sequential(
#             nn.Linear(3072, 2048),
#             nn.BatchNorm1d(2048),
#             nn.ReLU(),
#             nn.Linear(2048, 5),
#             nn.BatchNorm1d(5),
#             nn.ReLU(),
#             nn.Softmax(dim=1) 
#         )  
#         self.classifier_fc_with_gaze = nn.Sequential(
#             nn.Linear(3136, 2048),
#             nn.BatchNorm1d(2048),
#             nn.ReLU(),
#             nn.Linear(2048, 5),
#             nn.BatchNorm1d(5),
#             nn.ReLU(),
#             nn.Softmax(dim=1) 
#         )  


#     def forward(self, inside, outside, gaze):
#         out = self.conv_block(outside)
#         out = out.view(out.size(0),-1)
#         combined = torch.cat((inside, out,gaze), dim=1)
#         if gaze == None:
#             x = self.classifier_fc(combined)
#         else:
#             x = self.classifier_fc_with_gaze(combined)
#         return x
    

# # class AttentionModule(nn.Module):
# #     def __init__(self, input_dim, output_dim):
# #         super(AttentionModule, self).__init__()
# #         self.input_dim = input_dim
# #         self.output_dim = output_dim
# #         self.attention_fc = nn.Linear(self.input_dim, self.output_dim)
        
# #     def forward(self, x):
# #         # x: [batch_size, n_directions, features]
# #         attention_weights = F.softmax(self.attention_fc(x), dim=1)
# #         # attention_weights: [batch_size, n_directions, 1]
# #         output = torch.sum(x * attention_weights, dim=1)
# #         # output: [batch_size, features]
# #         return output, attention_weights
    
    

# class GazePredictorLSTM(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers, num_classes,drop_rate):
#         super(GazePredictorLSTM, self).__init__()
#         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
#         self.dropout = nn.Dropout(drop_rate)
#         self.fc = nn.Linear(hidden_dim, num_classes)
#         self.bn = nn.BatchNorm1d(num_classes)
    
#     def forward(self, x):
#         # LSTM은 입력 x와 함께 초기 hidden state(h_0)와 cell state(c_0)를 필요로 함
#         # h_0와 c_0는 default로 0으로 설정되어 있음
#         # x: (batch_size, seq_length, input_size)
#         out_not_fc, (h_n, c_n) = self.lstm(x)
        
#         # 마지막 타임 스텝의 히든 스테이트를 선형 레이어로 전달
#         out_not_fc = out_not_fc[:, -1, :]
#         out_not_fc = self.dropout(out_not_fc)
#         out = self.fc(out_not_fc)
#         out = self.bn(out)
#         return out, out_not_fc
    


# if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     torch.autograd.set_detect_anomaly(True)                   #nan발생시 검출
#     writer = SummaryWriter()
    
    
#     # inside 훈련 데이터 증강 및 데이터 로더 부분
#     opt = parse_opts()

    
#     opt.scales_inside = [opt.initial_scale]
#     for i in range(1, opt.n_scales_inside):
#         opt.scales_inside.append(opt.scales_inside[-1] * opt.scale_step)
#     opt.arch_inside = '{}-{}'.format(opt.model, opt.model_depth)
#     opt.mean_inside = get_mean(opt.norm_value_inside, dataset=opt.mean_dataset)
#     opt.std_inside = get_std(opt.norm_value_inside)
#     torch.manual_seed(opt.manual_seed)

#     if opt.no_mean_norm and not opt.std_norm:
#             norm_method = Normalize([0, 0, 0], [1, 1, 1])
#     elif not opt.std_norm:
#         norm_method = Normalize(opt.mean_inside, [1, 1, 1])
#     else:
#         norm_method = Normalize(opt.mean_inside, opt.std_inside)
          
#     if not opt.no_train_inside:
#         assert opt.train_crop in ['random', 'corner', 'center', 'driver focus']
#         if opt.train_crop == 'random':
#             crop_method = MultiScaleRandomCrop(opt.scales_inside, opt.sample_size)
#         elif opt.train_crop == 'corner':
#             crop_method = MultiScaleCornerCrop(opt.scales_inside, opt.sample_size)
#         elif opt.train_crop == 'center':
#             crop_method = MultiScaleCornerCrop(
#                 opt.scales_inside, opt.sample_size, crop_positions=['c'])
#         elif opt.train_crop == 'driver focus':
#             crop_method = DriverFocusCrop(opt.scales_inside, opt.sample_size)
#         train_spatial_transform = Compose([
#             crop_method,
#             MultiScaleRandomCrop(opt.scales_inside, opt.sample_size),
#             ToTensor(opt.norm_value_inside), norm_method
#         ])
        
#         # 랜덤하게 프레임을 잘라서 시간적으로도 crop하여 시간적인 데이터 증강을한다.
#         train_temporal_transform = UniformRandomSample(opt.sample_duration_inside, opt.end_second)
#         train_target_transform = ClassLabel()
#         train_horizontal_flip = RandomHorizontalFlip()
#         training_data_inside = get_training_set_inside(opt, train_spatial_transform, train_horizontal_flip,
#                                             train_temporal_transform, train_target_transform)
#         train_loader = torch.utils.data.DataLoader(
#             training_data_inside,
#             batch_size=opt.batch_size,
#             shuffle=True,
#             num_workers=opt.n_threads,
#             pin_memory=True,
#             drop_last=True)
#         train_logger_inside = Logger(
#             os.path.join(opt.result_path_inside, 'train_inside.log'),
#             ['epoch', 'loss', 'acc', 'lr'])
#         train_batch_logger_inside = Logger(
#             os.path.join(opt.result_path_inside, 'train_inside_batch.log'),
#             ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
#         train_logger_classifier = Logger(
#             os.path.join(opt.result_path_outside, 'train_classifier.log'),
#             ['epoch', 'loss', 'acc'])
        
        
#     # inside 검증 데이터 증강 및 데이터 로더 부분
#     if not opt.no_val_inside:
#         val_spatial_transform = Compose([
#                 DriverCenterCrop(opt.scales_inside, opt.sample_size),
#                 ToTensor(opt.norm_value_inside), norm_method
#             ])
#         val_temporal_transform = UniformEndSample(opt.sample_duration_inside, opt.end_second)
#         val_target_transform = ClassLabel()
#         validation_data = get_validation_set_inside(
#             opt, val_spatial_transform, val_temporal_transform, val_target_transform)
#         val_loader = torch.utils.data.DataLoader(
#             validation_data,
#             batch_size=opt.batch_size,
#             shuffle=False,
#             num_workers=opt.n_threads,
#             pin_memory=True,
#             drop_last=True)
#         val_logger_inside = Logger(
#             os.path.join(opt.result_path_inside, 'val_inside.log'), ['epoch', 'loss', 'acc'])
#         val_logger_classifier = Logger(
#             os.path.join(opt.result_path_outside, 'val_classifier.log'), ['epoch', 'loss', 'acc'])
    
    

#     # inside 모델 불러오기
#     model_inside, parameters_inside = generate_model(opt)
#     weights = [1, 2, 4, 2, 4]
#     class_weights = torch.FloatTensor(weights).cuda()
#     criterion_inside = nn.CrossEntropyLoss(weight=class_weights)
#     if not opt.no_cuda:
#         criterion_inside = criterion_inside.cuda()
        
#     if opt.nesterov_inside:
#         dampening = 0
#     else:
#         dampening = opt.dampening
    
#     optimizer_inside = optim.SGD(
#         parameters_inside,
#         lr=opt.learning_rate,
#         momentum=opt.momentum,
#         dampening=dampening,
#         weight_decay=opt.weight_decay,
#         nesterov=opt.nesterov_inside)
    
#     scheduler_inside = lr_scheduler.MultiStepLR(
#         optimizer_inside, milestones=opt.lr_step, gamma=0.1)
    
#     if opt.resume_path_inside:
#         print('loading checkpoint {}'.format(opt.resume_path_inside))
#         checkpoint = torch.load(opt.resume_path_inside)
#         assert opt.arch_inside == checkpoint['arch']

#         opt.begin_epoch = checkpoint['epoch']
#         model_inside.load_state_dict(checkpoint['state_dict'])
#         if not opt.no_train_inside:
#             optimizer_inside.load_state_dict(checkpoint['optimizer'])
            
    
    
    
#     # outside 훈련 데이터 증강 및 데이터 로더 부분
#     opt.scales_outside = [opt.initial_scale]
#     for i in range(1, opt.n_scales_outside):
#         opt.scales_outside.append(opt.scales_outside[-1] * opt.scale_step)
#     opt.arch_outside = 'ConvLSTM'
#     opt.mean_outside = get_mean(opt.norm_value_outside, dataset=opt.mean_dataset)
#     opt.std_outside = get_std(opt.norm_value_outside)
    
#     if opt.no_mean_norm and not opt.std_norm:
#             norm_method = Normalize([0, 0, 0], [1, 1, 1])
#     elif not opt.std_norm:
#         norm_method = Normalize(opt.mean_outside, [1, 1, 1])
#     else:
#         norm_method = Normalize(opt.mean_outside, opt.std_outside)
        
#     if not opt.no_train_outside:    
#         assert opt.train_crop in ['random', 'corner', 'center', 'driver focus']
#     if opt.train_crop == 'random':
#         crop_method = MultiScaleRandomCrop(opt.scales_outside, opt.sample_size)
#     elif opt.train_crop == 'corner':
#         crop_method = MultiScaleCornerCrop(opt.scales_outside, opt.sample_size)
#     elif opt.train_crop == 'center':
#         crop_method = MultiScaleCornerCrop(
#             opt.scales_outside, opt.sample_size, crop_positions=['c'])
#     elif opt.train_crop == 'driver focus':
#         crop_method = DriverFocusCrop(opt.scales_outside, opt.sample_size)
#     train_spatial_transform = Compose([
#         Scale(opt.sample_size),		
#         ToTensor(opt.norm_value_outside) #, norm_method
#     ])
#     train_temporal_transform = UniformIntervalCrop(opt.sample_duration_outside, opt.interval)
#     train_target_transform = Compose([
#         Scale(opt.sample_size),
#         ToTensor(opt.norm_value_outside)#, norm_method
#     ])
#     train_horizontal_flip = RandomHorizontalFlip()
#     training_data_outside = get_training_set_outside(opt, train_spatial_transform, train_horizontal_flip,
#                                         train_temporal_transform, train_target_transform)
#     train_loader_outside = torch.utils.data.DataLoader(
#             training_data_outside,
#             batch_size=opt.batch_size,
#             shuffle=True,
#             num_workers=opt.n_threads,
#             pin_memory=True,
#             drop_last=True)
#     train_logger_outside = Logger(
# 		os.path.join(opt.result_path_outside, 'train_outside.log'),
# 		['epoch', 'loss', 'lr'])
#     train_batch_logger_outside = Logger(
#         os.path.join(opt.result_path_outside, 'train_outside_batch.log'),
#         ['epoch', 'batch', 'iter', 'loss', 'lr'])
    
    
    
    
#     # outside 검증 데이터 증강 및 데이터 로더 부분
#     if not opt.no_val_outside:
#         val_spatial_transform = Compose([
#                 Scale(opt.sample_size),
#                 ToTensor(opt.norm_value_outside)#, norm_method
#             ])
#         val_temporal_transform = UniformIntervalCrop(opt.sample_duration_outside, opt.interval)
#         val_target_transform = val_spatial_transform
#         validation_data = get_validation_set_outside(
#             opt, val_spatial_transform, val_temporal_transform, val_target_transform)
#         val_loader_outside = torch.utils.data.DataLoader(
#             validation_data,
#             batch_size=opt.batch_size,
#             shuffle=False,
#             num_workers=opt.n_threads,
#             pin_memory=True,
#             drop_last=True)
#         val_logger_outside = Logger(
#             os.path.join(opt.result_path_outside, 'val_outside.log'), ['epoch', 'loss'])
    
#     # outside 모델 불러오기
#     model_outside = encoder(hidden_channels=[128, 64, 64, 32], sample_size=opt.sample_size, sample_duration=opt.sample_duration_outside).cuda()
	
#     model_outside = nn.DataParallel(model_outside, device_ids=None)
#     parameters_outside = model_outside.parameters()
    
#     criterion_outside = nn.MSELoss()
#     if not opt.no_cuda:
#         criterion_outside = criterion_outside.cuda()
        
#     if opt.nesterov_outside:
#         dampening = 0
#     else:
#         dampening = opt.dampening
#     optimizer_outside = optim.SGD(
# 		parameters_outside,
# 		lr=opt.learning_rate,
# 		momentum=opt.momentum,
# 		dampening=dampening,
# 		weight_decay=opt.weight_decay,
# 		nesterov=opt.nesterov_outside)
#     scheduler_outside = lr_scheduler.MultiStepLR(
#         optimizer_outside, milestones=opt.lr_step, gamma=0.1)
    
    
#     if opt.resume_path_outside:
#         print('loading checkpoint {}'.format(opt.resume_path_outside))
#         checkpoint = torch.load(opt.resume_path_outside)
#         assert opt.arch_outside == checkpoint['arch']

#         opt.begin_epoch = checkpoint['epoch']
#         model_outside.load_state_dict(checkpoint['state_dict'])
#         if not opt.no_train_outside:
#             optimizer_outside.load_state_dict(checkpoint['optimizer'])



#     # gaze Train dataloader
#     training_data_gaze = get_training_set_gaze(opt)
#     train_loader_gaze = torch.utils.data.DataLoader(
#         training_data_gaze,
#         batch_size=opt.batch_size,
#         shuffle=True,
#         num_workers=opt.n_threads,
#         pin_memory=True,
#         drop_last=True
#     )
#     train_batch_logger_gaze = Logger(
#     os.path.join(opt.result_path_inside, 'train_batch_gaze.log'),
#     ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])

#     # Gaze 모델의 에폭별 로깅을 위한 로거 초기화
#     train_logger_gaze = Logger(
#         os.path.join(opt.result_path_inside, 'train_gaze.log'),
#         ['epoch', 'loss', 'acc', 'lr'])
    

#     # Gaze Validation dataloader
#     val_data_gaze = get_validation_set_gaze(opt)
#     val_loader_gaze = torch.utils.data.DataLoader(
#         val_data_gaze,
#         batch_size=opt.batch_size,
#         shuffle=True,
#         num_workers=opt.n_threads,
#         pin_memory=True,
#         drop_last=True
#     )
#     # Gaze 모델의 에폭별 로깅을 위한 로거 초기화
#     val_logger_gaze = Logger(
#         os.path.join(opt.result_path_inside, 'val_gaze.log'),
#         ['epoch', 'loss', 'acc'])


#     #Gaze 모델 불러오기
#     gaze_predictor = GazePredictorLSTM(input_dim=17, hidden_dim=64, num_layers=2, num_classes=5, drop_rate=0.5).to(device)
#     gaze_predictor = nn.DataParallel(gaze_predictor, device_ids=None)

#     criterion_gaze = nn.CrossEntropyLoss()
#     optimizer_gaze = torch.optim.Adam(gaze_predictor.parameters(), lr=1e-4)

#     def calculate_metrics(output, target):
#         _, predicted = torch.max(output.data, 1)
#         correct = (predicted == target).sum().item()
#         total = target.size(0)
#         accuracy = correct / total
#         # 여기에 정밀도, 재현율, F1 점수 등 추가 계산 가능
#         return accuracy
    




#     My_Conv_classifier = conv_classifier().to(device)
    
#     weights = [1, 2, 4, 2, 4]
#     print(torch.cuda.is_available())
#     class_weights = torch.FloatTensor(weights).cuda()
#     criterion_classifier = nn.CrossEntropyLoss(weight=class_weights)
#     optimizer_classifier = torch.optim.Adam(My_Conv_classifier.parameters(), lr=0.001)
    
#     if not opt.no_cuda:
#         criterion = criterion_classifier.cuda()
        
#     global best_prec_inside
#     global best_loss_outside
#     global best_prec_classifier
#     best_prec_inside = 0
#     best_loss_outside = 100
#     best_prec_classifier = 0
    
#     iter_loader_gaze = iter(train_loader_gaze)

#     for epoch in range(opt.n_epochs + 1):  
#         torch.backends.cudnn.enabled=False
#         avg_acc = []
#         avg_loss = [] 
#         avg_acc_gaze = []
#         avg_loss_gaze = []         
        
#         if not opt.no_train_inside and not opt.no_train_outside:
#             # print('train_inside at epoch {}'.format(epoch))
            
#             losses_inside_train = AverageMeter()
#             accuracies_inside_train = AverageMeter()
            
#             # print('train_outside at epoch {}'.format(epoch))
            
#             losses_outside_train = AverageMeter()
#             losses_gaze_train = AverageMeter()
#             accuracies_gaze_train = AverageMeter()
            
#             # print('train_classifier at epoch {}'.format(epoch))
            
            
#             data_loader_train = tqdm(zip(train_loader, train_loader_outside), total=len(train_loader), desc = "Training")
#             # inside, outside 훈련 과정 실행
#             for i, ((inputs_in, targets_in), (inputs_out, targets_out)) in enumerate(data_loader_train):

#                 #Because of the diffenrence between in,out dataloader and gaze dataloader
#                 try:
#                     inputs_gaze, targets_gaze = next(iter_loader_gaze)
#                 except StopIteration:
#                     # Gaze 데이터셋의 끝에 도달했을 경우, 반복자를 다시 생성
#                     iter_loader_gaze = iter(train_loader_gaze)
#                     inputs_gaze, targets_gaze = next(iter_loader_gaze)

                
#                 # inside 훈련 과정 실행
#                 model_inside.train()
#                 if not opt.no_cuda:
#                     targets = targets_in.cuda(non_blocking=True)
#                 inputs_in = Variable(inputs_in)
#                 targets_in = Variable(targets_in)
#                 targets_in = targets_in.to(device) # inputs_in은 GPU인데 targets_in은 CPU라 오류떠서 해줘야돼
                
#                 outputs_in, outputs_in_not_fc = model_inside(inputs_in)
#                 loss_in = criterion_inside(outputs_in, targets_in)
#                 acc_in = calculate_accuracy(outputs_in, targets_in)
                
                

                
#                 # outside 훈련 과정 실행
#                 model_outside.train()
#                 """
#                 Consider only flir4(forward) direction
#                 """
#                 if not opt.no_cuda:
#                     targets_out = targets_out['flir4'].cuda(non_blocking=True)

#                 # inputs_out = Variable(inputs_out)
#                 #targets_out = Variable(targets_out)
                    
#                 inputs_out = inputs_out['flir4'].to(device)
#                 targets_out = targets_out.to(device)
                
#                 outputs_out = model_outside(inputs_out)
#                 loss_out = criterion_outside(outputs_out, targets_out)
                

#                 """
#                 Simply Consider 4 direction
#                 """

#                 # 각 방향별 훈련 결과와 가중치 적용

                
#                 # weighted_outputs = []
#                 # total_loss_out =0
#                 # for direction in ['flir4', 'flir1', 'flir2', 'flir3']:
#                 #     if not opt.no_cuda:
#                 #         targets_out_no = targets_out[direction].cuda(non_blocking=True)
#                 #     inputs_outs = inputs_out[direction].to(device)
#                 #     targets_outs = targets_out_no.to(device)
                    
#                 #     # 각 방향별 모델 실행
#                 #     outputs_out = model_outside(inputs_outs)
#                 #     loss_out = criterion_outside(outputs_out, targets_outs)
#                 #     if direction == 'flir4':
#                 #         total_loss_out += loss_out * 2
#                 #     else:
#                 #         total_loss_out+= loss_out
                    
                    
#                 #     weighted_outputs.append(outputs_out)

#                 # # 가중치가 적용된 출력을 합침
#                 # outputs_out_combined = torch.sum(torch.stack(weighted_outputs), dim=0)

                


#                 """
#                 Consider 4 direction using Attention module
#                 """

#                 #어텐션 모듈 초기화
#                 # attention_module = AttentionModule(input_dim=512, output_dim=1).to(device)

#                 # # 모델 훈련 과정에서 각 방향별 출력 준비
#                 # # 예를 들어, outputs_out이 각 방향별 출력을 포함하는 딕셔너리라고 가정
#                 # direction_outputs = [outputs_out[direction] for direction in ['flir4', 'flir1', 'flir2', 'flir3']]
#                 # direction_outputs_stack = torch.stack(direction_outputs, dim=1)
#                 # # direction_outputs_stack: [batch_size, n_directions, features]

#                 # # 어텐션 적용
#                 # outputs_out_combined, attention_weights = attention_module(direction_outputs_stack)
                

#                 """
#                 Training Gazepoint
#                 """

#                 # gaze 훈련 과정 실행
#                 inputs_gaze = inputs_gaze.to(device)
#                 targets_gaze = targets_gaze.to(device)  # 타겟 클래스 인덱스가 제공된다고 가정
#                 outputs_gaze,outputs_gaze_not_fc = gaze_predictor(inputs_gaze)
#                 loss_gaze = criterion_gaze(outputs_gaze, targets_gaze)
#                 acc_gaze = calculate_accuracy(outputs_gaze, targets_gaze)
#                 avg_acc_gaze.append(acc_gaze)
#                 avg_loss_gaze.append(loss_gaze.item())






                
#                 # classifier 훈련 과정 실행
#                 My_Conv_classifier.train()
#                 output = My_Conv_classifier(outputs_in_not_fc, outputs_out, outputs_gaze_not_fc)
#                 # Simply Consider 4 direction
#                 # output = My_Conv_classifier(outputs_in_not_fc, outputs_out_combined)
#                 # output = My_Conv_classifier(outputs_in_not_fc, outputs_out_combined)
#                 loss_classifier = criterion_classifier(output, targets_in)
#                 acc = calculate_accuracy(output, targets_in)
#                 avg_acc.append(acc)
#                 avg_loss.append(loss_classifier)
                
#                 # loss update
#                 losses_inside_train.update(loss_in.item(), inputs_in.size(0))
#                 losses_outside_train.update(loss_out.item(), inputs_out.size(0))
#                 losses_gaze_train.update(loss_gaze.item(), inputs_gaze.size(0))
#                 accuracies_gaze_train.update(acc_gaze, inputs_gaze.size(0))
                
#                 accuracies_inside_train.update(acc_in, inputs_in.size(0))

#                 """
#                 Calculate loss plus
#                 """
#                 # total_loss = loss_in + loss_out + loss_gaze + loss_classifier
                
#                 # optimizer update
#                 optimizer_inside.zero_grad()
#                 optimizer_outside.zero_grad()
#                 optimizer_gaze.zero_grad()
#                 optimizer_classifier.zero_grad()
                
#                 """
#                 Calculate each loss 
#                 """
#                 loss_in.backward(retain_graph=True)
#                 loss_out.backward(retain_graph=True)
#                 loss_gaze.backward(retain_graph=True)
#                 loss_classifier.backward()

#                 # total_loss.backward()

                

                
#                 optimizer_inside.step()
#                 optimizer_outside.step()
#                 optimizer_gaze.step()
#                 optimizer_classifier.step()
               
#                 writer.add_scalar('Training Loss Inside', losses_inside_train.avg, epoch)
#                 writer.add_scalar('Training Accuracy Inside', accuracies_inside_train.avg, epoch)
                
#                 writer.add_scalar('Training Loss Outside', losses_outside_train.avg, epoch)

                

#                 data_loader_train.set_postfix(loss_gaze=losses_gaze_train.val, acc_gaze=accuracies_gaze_train.val)
#                 data_loader_train.set_postfix(loss=loss_classifier.item(), acc=acc)
                
#                 # logger update
#                 train_batch_logger_inside.log({
#                     'epoch': epoch,
#                     'batch': i + 1,
#                     'iter': (epoch - 1) * len(train_loader) + (i + 1),
#                     'loss': losses_inside_train.val,
#                     'acc': accuracies_inside_train.val,
#                     'lr': optimizer_inside.param_groups[0]['lr']
#                 })
#                 train_batch_logger_outside.log({
#                     'epoch': epoch,
#                     'batch': i + 1,
#                     'iter': (epoch - 1) * len(train_loader_outside) + (i + 1),
#                     'loss': losses_outside_train.val,
#                     'lr': optimizer_outside.param_groups[0]['lr']
#                 })
#                 train_batch_logger_gaze.log({
#                 'epoch': epoch,
#                 'batch': i + 1,
#                 'iter': (epoch - 1) * len(train_loader_gaze) + (i + 1),
#                 'loss': losses_gaze_train.val,
#                 'acc': accuracies_gaze_train.val,
#                 'lr': optimizer_gaze.param_groups[0]['lr']
#             })
                
                
                
#             print('Epoch_inside: [{0}][{1}/{2}]\t'
#                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                         'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
#                             epoch,
#                             i + 1,
#                             len(train_loader),
#                             loss=losses_inside_train,
#                             acc=accuracies_inside_train))
#             print('Epoch_outside: [{0}][{1}/{2}]\t'
#                             'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
#                                 epoch,
#                                 i + 1,
#                                 len(train_loader_outside),
#                                 loss=losses_outside_train))
            
#             avg_acc_value_gaze = sum(avg_acc_gaze) / len(avg_acc_gaze)
#             avg_loss_value_gaze = sum(avg_loss_gaze) / len(avg_loss_gaze)
#             print(f"Epoch_gaze [{epoch}/{opt.n_epochs}], Loss: {avg_loss_value_gaze}, Avg Acc: {avg_acc_value_gaze}")
            
#             avg_acc_value_train = sum(avg_acc) / len(avg_acc)
#             avg_loss_value_train = sum(avg_loss) / len(avg_loss)
#             print(f"Epoch_classifier [{epoch}/{opt.n_epochs}], Loss: {avg_loss_value_train}, avg_acc: {avg_acc_value_train}")
            
#             train_logger_inside.log({
#                 'epoch': epoch,
#                 'loss': losses_inside_train.avg,
#                 'acc': accuracies_inside_train.avg,
#                 'lr': optimizer_inside.param_groups[0]['lr']
#             })
#             train_logger_classifier.log({
#                     'epoch': epoch,
#                     'loss': avg_loss_value_train,
#                     'acc': avg_acc_value_train
#                 })
#             train_logger_gaze.log({
#             'epoch': epoch,
#             'loss': losses_gaze_train.avg,
#             'acc': accuracies_gaze_train.avg,
#             'lr': optimizer_gaze.param_groups[0]['lr']
#         })

#             if epoch % opt.checkpoint == 0:
#                 save_file_path_inside = os.path.join(opt.result_path_inside,
#                                             'save_{}.pth'.format(epoch))
#                 states = {
#                     'epoch': epoch + 1,
#                     'arch': opt.arch_inside,
#                     'state_dict': model_inside.state_dict(),
#                     'optimizer': optimizer_inside.state_dict(),
#                 }
                
#             train_logger_outside.log({
#                 'epoch': epoch,
#                 'loss': losses_outside_train.avg,
#                 'lr': optimizer_outside.param_groups[0]['lr']
#             })

#             if epoch % opt.checkpoint == 0:
#                 save_file_path_outside = os.path.join(opt.result_path_outside,
#                                                 'convlstm-save_{}.pth'.format(epoch))
#                 states = {
#                     'epoch': epoch + 1,
#                     'arch': opt.arch_outside,
#                     'state_dict': model_outside.state_dict(),
#                     'optimizer': optimizer_outside.state_dict(),
#                 }  
#             writer.add_scalar('Training Loss Classifier', avg_acc_value_train, epoch)
#             writer.add_scalar('Training Accuracy Classifier', avg_loss_value_train, epoch)
#         ################################################################################################################
        
#         avg_acc_val = []
#         avg_loss_val = []
#         avg_acc_gaze_val = []
#         avg_loss_gaze_val = []
#         if not opt.no_val_inside and not opt.no_val_outside:
#             # print('validation_inside at epoch {}'.format(epoch))
#             model_inside.eval()
#             losses_inside = AverageMeter()
#             accuracies_inside = AverageMeter()
            
#             # print('validation_outside at epoch {}'.format(epoch))
#             model_outside.eval()
#             losses_outside = AverageMeter()

#             gaze_predictor.eval()
#             losses_gaze_val = AverageMeter()
#             accuracies_gaze_val = AverageMeter()
            
#             # print('validation_classifier at epoch {}'.format(epoch))
#             My_Conv_classifier.eval()
            
#             data_loader_val = tqdm(zip(val_loader, val_loader_outside), total=len(val_loader), desc = "Validation")
#             # inside, outside 검증 과정 실행

#             iter_loader_gaze = iter(val_loader_gaze)

#             with torch.no_grad():
#                 for i, ((inputs_in, targets_in), (inputs_out, targets_out)) in enumerate(data_loader_val):

#                     try:
#                         inputs_gaze, targets_gaze = next(iter_loader_gaze)
#                     except StopIteration:
#                         # Gaze 데이터셋의 끝에 도달했을 경우, 반복자를 다시 생성하고 계속 진행
#                         iter_loader_gaze = iter(val_loader_gaze)
#                         inputs_gaze, targets_gaze = next(iter_loader_gaze)


#                     # inside 검증 과정 실행
#                     if not opt.no_cuda:
#                         targets = targets_in.cuda(non_blocking=True)
#                     inputs_in = inputs_in.to(device)
#                     targets_in = targets_in.to(device) # inputs_in은 GPU인데 targets_in은 CPU라 오류떠서 해줘야돼
                    
#                     outputs_in, outputs_in_not_fc = model_inside(inputs_in)
#                     loss_in = criterion_inside(outputs_in, targets_in)
#                     acc_in = calculate_accuracy(outputs_in, targets_in)
                    
#                     # outside 검증 과정 실행
#                     if not opt.no_cuda:
#                         targets_out = targets_out['flir4'].cuda(non_blocking=True)

#                     inputs_out = inputs_out['flir4'].to(device)
#                     targets_out = targets_out.to(device)
                    
#                     outputs_out = model_outside(inputs_out)
#                     loss_out = criterion_outside(outputs_out, targets_out)


#                     # Gaze 모델의 검증 과정
#                     inputs_gaze = inputs_gaze.to(device)
#                     targets_gaze = targets_gaze.to(device)
#                     outputs_gaze, outputs_gaze_not_fc = gaze_predictor(inputs_gaze)
#                     loss_gaze = criterion_gaze(outputs_gaze, targets_gaze)
#                     acc_gaze = calculate_accuracy(outputs_gaze, targets_gaze)
#                     avg_acc_gaze_val.append(acc_gaze)
#                     avg_loss_gaze_val.append(loss_gaze.item())
                    
#                     # classifier 검증 과정 실행
#                     output = My_Conv_classifier(outputs_in_not_fc, outputs_out, outputs_gaze_not_fc)
#                     loss_classifier = criterion_classifier(output, targets_in)
#                     acc = calculate_accuracy(output, targets_in)
#                     avg_acc_val.append(acc)
#                     avg_loss_val.append(loss_classifier)
                    
#                     # loss update
#                     losses_inside.update(loss_in.item(), inputs_in.size(0))
#                     losses_outside.update(loss_out.item(), inputs_out.size(0))
#                     losses_gaze_val.update(loss_gaze.item(), inputs_gaze.size(0))
#                     accuracies_gaze_val.update(acc_gaze, inputs_gaze.size(0))
#                     accuracies_inside.update(acc_in, inputs_in.size(0))


#                     writer.add_scalar('Validation Loss Inside', losses_inside.avg, epoch)
#                     writer.add_scalar('Validation Accuracy Inside', accuracies_inside.avg, epoch)
#                     writer.add_scalar('Validation Loss Outside', losses_outside.avg, epoch)
#                     writer.add_scalar('Validation Loss Gaze', losses_gaze_val.avg, epoch)
#                     writer.add_scalar('Validation Accuracy Gaze', accuracies_gaze_val.avg, epoch)
                    
#                     data_loader_val.set_postfix(loss=loss_classifier.item(), acc=acc)
                
                
#             print('Epoch_inside: [{0}][{1}/{2}]\t'
#                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                         'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
#                             epoch,
#                             i + 1,
#                             len(val_loader),
#                             loss=losses_inside,
#                             acc=accuracies_inside))
#             print('Epoch_outside: [{0}][{1}/{2}]\t'
#                             'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
#                                 epoch,
#                                 i + 1,
#                                 len(val_loader_outside),
#                                 loss=losses_outside))
#             avg_acc_gaze_value = sum(avg_acc_gaze_val) / len(avg_acc_gaze_val)
#             avg_loss_gaze_value = sum(avg_loss_gaze_val) / len(avg_loss_gaze_val)
#             print(f"Epoch_gaze: Gaze Loss: {avg_loss_gaze_value}, Gaze Accuracy: {avg_acc_gaze_value}")
            
#             avg_acc_value = sum(avg_acc_val) / len(avg_acc_val)
#             avg_loss_value = sum(avg_loss_val) / len(avg_loss_val)
#             print(f"Epoch_classifier [{epoch}/{opt.n_epochs}], Loss: {avg_loss_value}, avg_acc: {avg_acc_value}")
            
#             val_logger_inside.log({
#                 'epoch': epoch,
#                 'loss': losses_inside.avg,
#                 'acc': accuracies_inside.avg,
#             })

#             val_logger_classifier.log({
#                         'epoch': epoch,
#                         'loss': avg_loss_value,
#                         'acc': avg_acc_value,
#                     })
            
#             val_logger_gaze.log({'epoch': epoch, 'loss': losses_gaze_val.avg, 'acc': accuracies_gaze_val.avg})


#             val_logger_outside.log({
#                 'epoch': epoch,
#                 'loss': losses_outside.avg
#                 })
#             writer.add_scalar('Training Loss Inside', avg_acc_value, epoch)
#             writer.add_scalar('Training Accuracy Inside', avg_loss_value, epoch)

#             is_best_inside = accuracies_inside.avg > best_prec_inside
#             best_prec_inside = max(accuracies_inside.avg, best_prec_inside)
#             if is_best_inside:
#                 print('\n The best inside prec is %.4f' % best_prec_inside)
#                 states = {
#                     'epoch': epoch + 1,
#                     'arch': opt.arch_inside,
#                     'state_dict': model_inside.state_dict(),
#                     'optimizer': optimizer_inside.state_dict(),
#                 }
#                 save_file_path_inside = os.path.join(opt.result_path_inside,
#                                     'save_best_inside.pth')
#                 torch.save(states, save_file_path_inside)
                
            
            

#             is_best_outside = losses_outside.avg < best_loss_outside
#             best_loss_outside = min(losses_outside.avg, best_loss_outside)
            
#             if is_best_outside:
#                 print('\n The best outside loss is %.4f' % best_loss_outside)
#                 states = {
#                     'epoch': epoch + 1,
#                     'arch': opt.arch_outside,
#                     'state_dict': model_outside.state_dict(),
#                     'optimizer': optimizer_outside.state_dict(),
#                 }
#                 save_file_path_outside = os.path.join(opt.result_path_outside,
#                                     'save_best_outside.pth')
#                 torch.save(states, save_file_path_outside)
                
            
#             is_best_classifier = avg_acc_value > best_prec_classifier
#             best_prec_classifier = max(avg_acc_value, best_prec_classifier)
            
#             if is_best_classifier:
#                 print('\n The best classifier prec is %.4f' % best_prec_classifier)
#                 states = {
#                     'epoch': epoch + 1,
#                     'state_dict': My_Conv_classifier.state_dict(),
#                     'optimizer': optimizer_classifier.state_dict(),
#                 }
#                 save_file_path_classifier = os.path.join(opt.result_path_outside,
#                                     'save_best_classifier.pth')
#                 torch.save(states, save_file_path_classifier)    
#         if not opt.no_train_inside and not opt.no_train_outside and not opt.no_val_inside and not opt.no_val_outside:
#             scheduler_inside.step()
#             scheduler_outside.step() 
#         torch.backends.cudnn.enabled=False
    
#     print(f'Test Accuracy: {best_prec_classifier:.4f}')
            

# ######################################################################################################################
    
#     def load_checkpoint(model, filename):
#         if os.path.isfile(filename):
#             print("=> loading checkpoint '{}'".format(filename))
#             checkpoint = torch.load(filename)
#             model.load_state_dict(checkpoint['state_dict'])
#             print("=> loaded checkpoint '{}'".format(filename))
#         else:
#             print("=> no checkpoint found at '{}'".format(filename))
#         return model


#     writer.close()