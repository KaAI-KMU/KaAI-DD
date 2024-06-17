import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from opts import parse_opts
from model import generate_model
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor, DriverFocusCrop, DriverCenterCrop)
from temporal_transforms import LoopPadding, TemporalRandomCrop, TemporalCenterCrop, UniformRandomSample, UniformEndSample
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset import get_training_set_inside, get_training_set_outside
from utils import Logger, AverageMeter, calculate_accuracy
from models.convolution_lstm import encoder
from mean import get_mean, get_std

def extract_inside_vector(inside_model_path, input_data):
    # 모델 및 옵션 설정
    
    model, _ = generate_model(opt)
    checkpoint = torch.load(inside_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()
    with torch.no_grad():
        inputs = input_data.float().to(device)
        outputs = model(inputs)
        inside_vector = outputs.cpu().numpy()

    return inside_vector

def extract_outside_vector(outside_model_path, input_data):
    model = encoder(hidden_channels=[128, 64, 64, 32], sample_size=opt.sample_size, sample_duration=opt.sample_duration_outside).to(device)    
    checkpoint = torch.load(outside_model_path)
    state_dict = checkpoint['state_dict'] 
    new_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    model.load_state_dict(new_state_dict)

    model.eval()
    with torch.no_grad():
        inputs = input_data.float().to(device)
        outputs = model(inputs) 
        # To Tensor
        outside_vector = outputs.cpu().numpy()

    return outside_vector

class Conv_Block(nn.Module):
    def __init__(self):
        super(Conv_Block, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            
        )
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=0)
            
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        #flatten the output
        x = x.view(x.size(0),-1)
        return x
    
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # FC0: 3072 -> 2048
        
        self.Classifier_fc = nn.Sequential(
            nn.Linear(3072, 2048),
            # nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 5),
            # nn.BatchNorm1d(5),
            nn.ReLU(),
            nn.Softmax(dim=1) 
        )

    def forward(self, x):
        x = self.Classifier_fc(x)
        return x

# 모델 학습 및 평가 함수
def train_model(model, criterion, optimizer, train_loader, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)

    return epoch_loss, epoch_acc

def evaluate_model(model, criterion, val_loader, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = running_corrects.double() / len(val_loader.dataset)

    return epoch_loss, epoch_acc

# 모델, 데이터 로더 및 기타 구성 요소 초기화
if __name__ == '__main__':
    opt = parse_opts()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inside_pth_path = './inresults/save_best.pth'
    outside_pth_path = './outresults/convlstm-save_best.pth'
    
    # opt.mean_outside = get_mean(opt.norm_value_outside, dataset=opt.mean_dataset)
    # opt.std_outside = get_std(opt.norm_value_outside)
    
    

    # Inside 및 Outside 모델 로드
    # inside_model = generate_model(opt).to(device)
    # outside_model = generate_model(opt).to(device) # 외부 모델의 구조가 다를 경우 해당 구조로 초기화 필요

    # ConvBlock 및 Classifier 초기화
    conv_block = Conv_Block().to(device)
    classifier = Classifier().to(device)

    # 훈련 inside 데이터 증강 및 데이터 로더 설정
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales_inside):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)  
    opt.mean = get_mean(opt.norm_value_inside, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value_inside)
    
    
    
    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)
    
    
    if not opt.no_train: 
        assert opt.train_crop in ['random', 'corner', 'center', 'driver focus']
        if opt.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(
                opt.scales, opt.sample_size, crop_positions=['c'])
        elif opt.train_crop == 'driver focus':
            crop_method = DriverFocusCrop(opt.scales, opt.sample_size)
        train_spatial_transform = Compose([
            crop_method,
            MultiScaleRandomCrop(opt.scales, opt.sample_size),
            ToTensor(opt.norm_value_inside), norm_method
        ])
        # 랜덤하게 프레임을 잘라서 시간적으로도 crop하여 시간적인 데이터 증강을한다.
        train_temporal_transform = UniformRandomSample(opt.sample_duration, opt.end_second)
        train_target_transform = ClassLabel()
        train_horizontal_flip = RandomHorizontalFlip()
        training_data = get_training_set_inside(opt, train_spatial_transform, train_horizontal_flip,
                                        train_temporal_transform, train_target_transform)
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)
    
    
    
    train_temporal_transform = UniformRandomSample(opt.sample_duration, opt.end_second)
    train_target_transform = ClassLabel()
    train_horizontal_flip = RandomHorizontalFlip()
    training_data_inside = get_training_set_inside(opt, train_spatial_transform, train_horizontal_flip,train_temporal_transform, train_target_transform)
    
    # 훈련 outside 데이터 증강 및 데이터 로더 설정
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
        
    opt.arch = 'ConvLSTM'
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    
    
    
    
    
    
    training_data_outside = get_training_set_outside(opt, spatial_transform, temporal_transform, target_transform)

    train_loader_inside = DataLoader(training_data_inside, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_threads)
    train_loader_outside = DataLoader(training_data_outside, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_threads)

    # 분류기 학습을 위한 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=opt.learning_rate, momentum=opt.momentum)

    best_acc = 0.0
    best_loss = float('inf')

    # 학습 및 평가 루프
    for epoch in range(opt.n_epochs):
        # Inside 및 Outside 데이터에 대한 특징 추출 및 ConvBlock 처리
        inside_features = extract_inside_vector(inside_pth_path, train_loader_inside, device)
        outside_features = extract_outside_vector(outside_pth_path, train_loader_outside, device)
        outside_features = conv_block(outside_features)

        # Inside 및 Outside 특징 결합
        combined_features = torch.cat((inside_features, outside_features), dim=1)

        # 분류기 학습
        train_loss, train_acc = train_model(classifier, criterion, optimizer, combined_features, device)

        # 검증 세트를 사용한 평가
        val_loss, val_acc = evaluate_model(classifier, criterion, combined_features, device)

        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # 최고 정확도 및 최소 손실 업데이트
        if val_acc > best_acc:
            best_acc = val_acc
            # 정확도가 개선되었을 때 모델 저장
            torch.save(classifier.state_dict(), 'best_accuracy_model.pth')

        if val_loss < best_loss:
            best_loss = val_loss
            # 손실이 감소했을 때 모델 저장
            torch.save(classifier.state_dict(), 'best_loss_model.pth')

    print(f'Best Accuracy: {best_acc:.4f}, Best Loss: {best_loss:.4f}')
