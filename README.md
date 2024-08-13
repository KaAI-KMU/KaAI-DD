# Driver-Intention-Prediction
A framework to predict driver's maneuver behaviors.

This repository presents a PyTorch implementation of the Driver Intent Prediction model from our paper "KaAI-DD: Holistic Driving Dataset for Predicting Driver Gaze and Intention". The framework builds upon and adapts techniques from the paper ["Driver Intention Anticipation Based on In-Cabin and Driving Scene Monitoring"] (https://arxiv.org/pdf/2006.11557.pdf). . We have modified the original implementation to allow benchmarking with the KaAI dataset, which we developed. Below, we explain how to test our dataset using this framework.

Here is a demo of our framework in action:  
![](demo.gif)

In this demo, predictions are made every second. If the prediction is correct, a ✓ appears.

## Dataset Preparation

We have prepared the KaAI 5s Dataset, which can be downloaded from [this link](https://drive.google.com/drive/u/1/folders/1ry2UwkBsjIKJkzwYr0E1XAaPtWdb7v8b). The dataset includes the following components:

1. **road_camera**: Videos recorded by a camera facing the road.
2. **face_camera**: Videos recorded by a camera facing the driver.
3. **Gaze & CAN**: Data containing the driver's gaze and vehicle CAN signals.

To use our dataset with this framework:

1. Place the `road_camera`, `face_camera`, and `Gaze&CAN` folders inside the `annotation_kaai` directory.
2. Split the dataset using 5-fold cross-validation by running the `n_fold_Brain4cars.py` script in the `datasets/annotation_kaai` directory.
   - You can also use the pre-generated `.csv` files available in the `datasets/annotation_kaai` directory to skip this step.

## Train/Evaluate 3D-ResNet50 with Inside Videos

The 3D-ResNet50 network, along with its pretrained model, is sourced from [3D ResNets](https://github.com/kenshohara/3D-ResNets-PyTorch). We express our gratitude to the authors of this project.

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

We used [FlowNet 2.0](https://github.com/NVIDIA/flownet2-pytorch) to extract the optical flow of all outside images. The optical flow images can also be found [here](https://bwstaff-my.sharepoint.com/:f:/g/personal/yao_rong_bwstaff_de/EpmuNb3eB7hPgv2DmeBrQ1ABqgQ6uInXudrpfQQyPgmJZA?e=RimExC).

Our ConvLSTM network is adapted from this [repo](https://github.com/automan000/Convolutional_LSTM_PyTorch). We extend our gratitude to the creators of these projects.

Before running the `run-ConvLSTM.sh` script, set the following paths:

1. **`root_path`**: Path to this project.
2. **`annotation_path`**: Path to the annotation directory in this project.
3. **`video_path`**: Path to the image frames of optical flow images.

**Important Notes**:

- **`n_fold`**: The fold number, ranging from 0 to 4.
- **`sample_duration`**: Length of input videos (5 frames).
- **`interval`**: Interval between frames in the input clip (between 5 and 30).
- **`end_second`**: The time before the maneuver from which frames are input (ranging from 1 to 5 seconds).

For more details, refer to Section IV.B of the [paper](https://arxiv.org/pdf/2006.11557.pdf).

## Citation

```
@INPROCEEDINGS{9294181,
  author={Rong, Yao and Akata, Zeynep and Kasneci, Enkelejda},
  booktitle={2020 IEEE 23rd International Conference on Intelligent Transportation Systems (ITSC)}, 
  title={Driver Intention Anticipation Based on In-Cabin and Driving Scene Monitoring}, 
  year={2020},
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/ITSC45102.2020.9294181}}
```
