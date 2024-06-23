---

The specified path contains a sample dataset and tracking code. Download the sample dataset and configure it according to each tracking path.

The code implements a multimodal object tracking system that combines image and LiDAR data, applying a Kalman filter and the Hungarian algorithm. In each frame, it loads image and LiDAR data and processes JSON files to obtain the position and size of objects. After setting the initial state of the objects, it estimates their position and velocity using the Kalman filter. ResNet-50 is used to extract feature vectors from the images, providing visual information. The system matches tracking targets using cost matrices based on Euclidean distance, cosine similarity, and IoU, and finds the optimal match using the Hungarian algorithm. It then updates the state through the Kalman filter, improving the accuracy and efficiency of real-time object tracking. The main techniques used are the Kalman filter, Hungarian algorithm, ResNet-50, and IoU.

**<3D_Tracking Process>**
1. Modify the paths for 3d_label, images, and LiDAR data.
   - The path at line 51 should contain the annotated JSON file (.json) for the point cloud.
   - The path at line 52 should contain the image files (.png) for the forward camera (flir4) of the dataset.
   - The path at line 53 should contain the LiDAR (.pcd) files of the dataset.
2. Modify the folder name in lines 207 and 208 to match the format 2024xxxx_drive_, adjusting the xxxx part as needed.
3. After making these modifications, run the Python file in the terminal.
4. Finally, validate the tracked parts in SustechPoint as per the annotation method mentioned above.

**<2D_Tracking Process>**
1. For 2D tracking, modify only the path at line 81.
   - This path should contain the 2d_label for the forward camera.
   - No other inputs are used.
2. Once the path is modified, run the Python file in the terminal to complete the process.

—_