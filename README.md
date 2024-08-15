# Driver-Intention-Prediction for KaAI-DD
A framework to predict driver's maneuver behaviors.

This repository presents a PyTorch implementation of the Driver Intent Prediction model from our paper ["KaAI-DD: Holistic Driving Dataset for Predicting Driver Gaze and Intention"](insert-link-to-your-paper).

Our framework builds on the foundational work of ["Driver Intention Anticipation Based on In-Cabin and Driving Scene Monitoring"](https://arxiv.org/pdf/2006.11557.pdf), with modifications and enhancements to benchmark performance using the KaAI dataset, which we developed. Below, we explain how to test our dataset using this framework.

![Driver Intent Prediction Model Architecture](https://github.com/user-attachments/assets/0b041f1e-afc6-480e-ac02-01a4278ca91b)  
<p align="center"><i>Architecture of our driver intent prediction model</i></p>

## Results

| Model          | Dataset                                                                 | Accuracy (%) |
|----------------|-------------------------------------------------------------------------|--------------|
| Baseline       | [Brain4Cars](https://arxiv.org/pdf/1601.00740) (In-Cabin)               | 77.40        |
| Baseline       | [Brain4Cars](https://arxiv.org/pdf/1601.00740) (In-Cabin + Out-Cabin)   | 83.98        |
| Baseline       | Our Dataset (In-Cabin)                                                  | 83.65        |
| Baseline       | Our Dataset (In-Cabin + Out-Cabin)                                      | 85.85        |
| **Enhanced**   | **Our Dataset (In-Cabin + Out-Cabin + Gaze)**                           | **86.85**    |


## Dataset Preparation

We have prepared the KaAI 5s Dataset, which can be downloaded from [this link](https://drive.google.com/drive/folders/1R4jd6eZb6MOGuZ-I4O60faz6FsRx9elY?usp=drive_link). The dataset includes the following components:

1. **road_camera**: Videos recorded by a camera facing the road.
2. **face_camera**: Videos recorded by a camera facing the driver.
3. **Gaze & CAN**: Data containing the driver's gaze and vehicle CAN signals.

To use our dataset with this framework:

1. Place the `road_camera`, `face_camera`, and `Gaze&CAN` folders inside the `annotation_kaai` directory.
2. Split the dataset using 5-fold cross-validation by running the `n_fold_Brain4cars.py` script in the `datasets/annotation_kaai` directory.
   - You can also use the pre-generated `.csv` files available in the `datasets/annotation_kaai` directory to skip this step.

## Train/Evaluate 3D-ResNet50 with Inside Videos

The 3D-ResNet50 network, along with its pretrained model, is adapted from the work in [3D ResNets](https://github.com/kenshohara/3D-ResNets-PyTorch). We express our gratitude to the original authors, and have modified the implementation to better suit our specific dataset and task.

Before running the `run-3DResnet.sh` script, set the following paths:

1. **`root_path`**: Path to this project.
2. **`annotation_path`**: Path to the annotation directory in this project.
3. **`video_path`**: Path to the image frames of driver videos.
4. **`pretrain_path`**: Path to the pretrained 3D ResNet50 model.

**Important Notes**:

- **`n_fold`**: The fold number, ranging from 0 to 4.
- **`sample_duration`**: Length of input videos (16 frames).
- **`end_second`**: The time before the maneuver from which frames are input (ranging from 1 to 5 seconds).

For more details on other arguments, refer to `opt.py`.

The model trained using our script is available [here](https://bwstaff-my.sharepoint.com/:f:/g/personal/yao_rong_bwstaff_de/EpmuNb3eB7hPgv2DmeBrQ1ABqgQ6uInXudrpfQQyPgmJZA?e=RimExC). The model name is `save_best_3DResNet50.pth`.

## Train/Evaluate ConvLSTM with Outside Videos

We utilized [FlowNet 2.0](https://github.com/NVIDIA/flownet2-pytorch) to extract the optical flow of all outside images, which we then used in our ConvLSTM network. The optical flow images can also be found [here](https://bwstaff-my.sharepoint.com/:f:/g/personal/yao_rong_bwstaff_de/EpmuNb3eB7hPgv2DmeBrQ1ABqgQ6uInXudrpfQQyPgmJZA?e=RimExC).

Our ConvLSTM network is adapted and extended from the work in this [repo](https://github.com/automan000/Convolutional_LSTM_PyTorch). We express our appreciation to the creators of these foundational projects.

Before running the `run-ConvLSTM.sh` script, set the following paths:

1. **`root_path`**: Path to this project.
2. **`annotation_path`**: Path to the annotation directory in this project.
3. **`video_path`**: Path to the image frames of optical flow images.

**Important Notes**:

- **`n_fold`**: The fold number, ranging from 0 to 4.
- **`sample_duration`**: Length of input videos (5 frames).
- **`interval`**: Interval between frames in the input clip (between 5 and 30).
- **`end_second`**: The time before the maneuver from which frames are input (ranging from 1 to 5 seconds).

