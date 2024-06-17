import numpy as np
import argparse
import os
import cv2
import pyrealsense2 as rs
import pandas as pd
from scipy.ndimage import zoom
import open3d as o3d
import matplotlib.pyplot as plt

def read_marker_info(txt_file):
    with open(txt_file, 'r') as file:
        markers = [list(map(float, line.strip().split())) for line in file.readlines()]
    if len(markers) >= 3:
        reordered_markers = [markers[1], markers[0], markers[2]]
    else:
        reordered_markers = markers
    return reordered_markers

def calculate_depth_adjustment(depth_map, marker1, marker2, radius):
    x1, y1 = int(marker1[0] * depth_map.shape[1]), int(marker1[1] * depth_map.shape[0])
    x2, y2 = int(marker2[0] * depth_map.shape[1]), int(marker2[1] * depth_map.shape[0])
    
    depth1 = depth_map[y1, x1]
    depth2 = depth_map[y2, x2]
    
    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y2) // 2
    mid_depth = (depth1 + depth2) / 2

    depth_diff = abs(depth1 - depth2)
    
    if radius > max(marker1[2], marker2[2]):
        adjusted_depth = mid_depth + depth_diff
    elif radius < min(marker1[2], marker2[2]):
        adjusted_depth = mid_depth - depth_diff
    else:
        adjusted_depth = mid_depth

    return adjusted_depth

def calculate_plane_normal(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    return normal / norm if norm else None

def draw_normal_vector(image, start_point, normal_vector, length=100):
    end_point = (int(start_point[0] + normal_vector[0] * length), 
                 int(start_point[1] + normal_vector[1] * length))
    cv2.arrowedLine(image, start_point, end_point, (0, 255, 0), 2)
    return image

def draw_depth_vector(image, start_point, z_depth, scale=100):
    end_point = (start_point[0], start_point[1] - int(z_depth * scale))
    cv2.arrowedLine(image, start_point, end_point, (255, 0, 0), 2)
    return image

def convert_depth_pixel_to_metric_coordinate(intrinsics, pixel, depth):
    return rs.rs2_deproject_pixel_to_point(intrinsics, pixel, depth)

def create_rotation_matrix(normal_vector):
    z_axis = normal_vector
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross([0, 1, 0], z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    
    rotation_matrix = np.array([x_axis, y_axis, z_axis])
    return rotation_matrix

def create_transformation_matrix(rotation_matrix, translation_vector):
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector
    return transformation_matrix

def apply_transformation(point, transformation_matrix):
    point_homogeneous = np.append(point, 1)
    transformed_point_homogeneous = np.dot(transformation_matrix, point_homogeneous)
    return transformed_point_homogeneous[:3]

def eyetracker_to_realsense(point):
    transformation_matrix = np.array([[1, 0, 0, 0],    # Eyetracker x -> RealSense x
                                      [0, 1, 0, 0],   # Eyetracker y -> RealSense y 
                                      [0, 0, 1, 0],    # Eyetracker z -> RealSense z
                                      [0, 0, 0, 1]])
    point_homogeneous = np.append(point, 1)
    transformed_point_homogeneous = np.dot(transformation_matrix, point_homogeneous)
    return transformed_point_homogeneous[:3]

def realsense_to_lidar(point):
    # LiDAR 좌표계의 12.5도 각도를 고려한 회전 변환 행렬
    angle = np.deg2rad(12.5)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ])
    
    transformation_matrix = np.array([
        [-1, 0, 0, 0],    # RealSense x -> LiDAR -x
        [0, 0, -1, 0],    # RealSense y -> LiDAR -z
        [0, -1, 0, 0],    # RealSense z -> LiDAR -y
        [0, 0, 0, 1]
    ])
    
    point_homogeneous = np.append(point, 1)
    transformed_point_homogeneous = np.dot(transformation_matrix, point_homogeneous)
    
    # 회전 변환 적용
    transformed_point = transformed_point_homogeneous[:3]
    rotated_point = np.dot(rotation_matrix, transformed_point)
    
    return rotated_point

def lidar_to_camera(lidar_point, camera_transform):
    point_homogeneous = np.append(lidar_point, 1)
    camera_point_homogeneous = np.dot(camera_transform, point_homogeneous)
    return camera_point_homogeneous[:3]

def project_to_image(camera_point, camera_intrinsics):
    x = (camera_intrinsics.fx * camera_point[0] / camera_point[2]) + camera_intrinsics.ppx
    y = (camera_intrinsics.fy * camera_point[1] / camera_point[2]) + camera_intrinsics.ppy
    return int(x), int(y)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker_folder", type=str, required=True, help="Path to the folder containing marker information")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the folder containing image files")
    parser.add_argument("--depth_folder", type=str, required=True, help="Path to the folder containing depth files")
    parser.add_argument("--lidar_folder", type=str, required=True, help="Path to the folder containing LiDAR point cloud files")
    parser.add_argument("--depth_format", type=str, choices=['npy', 'png'], required=True, help="Format of the depth files (npy or png)")
    parser.add_argument("--gaze_file", type=str, required=True, help="Path to the gaze data CSV file")
    parser.add_argument("--visualize", action='store_true', help="Visualize depth vectors on the images")
    parser.add_argument("--visualize_3d", action='store_true', help="Visualize 3D point cloud data")
    parser.add_argument("--frame", type=int, required=False, help="Specific frame number to process")
    parser.add_argument("--front_image", type=str, required=False, help="Path to the 2D image for projection")
    args = parser.parse_args()
    
    marker_files = sorted(os.listdir(args.marker_folder))
    initial_frame_number = int(marker_files[0].split('.')[0])
    
    frame_number = initial_frame_number

    # Manually set the depth camera intrinsics
    depth_intrinsics = rs.intrinsics()
    depth_intrinsics.width = 1280
    depth_intrinsics.height = 720
    depth_intrinsics.ppx = 638.51171875
    depth_intrinsics.ppy = 358.90625
    depth_intrinsics.fx = 622.219543457031
    depth_intrinsics.fy = 622.219543457031
    depth_intrinsics.model = rs.distortion.brown_conrady
    depth_intrinsics.coeffs = [0, 0, 0, 0, 0]

    # Load gaze data
    gaze_data = pd.read_csv(args.gaze_file)

    # Prepare list to store the results
    results = []

    # Camera extrinsics
    camera_extrinsics = np.array([
        [1, 0, 0, 0],      # LiDAR x -> Camera x
        [0, 0, -1, 0.223], # LiDAR y -> Camera -z (with translation)
        [0, 1, 0, -0.313], # LiDAR z -> Camera y (with translation)
        [0, 0, 0, 1]
    ])

    # Camera intrinsics
    camera_intrinsics = rs.intrinsics()
    camera_intrinsics.width = 808
    camera_intrinsics.height = 620
    camera_intrinsics.ppx = 404
    camera_intrinsics.ppy = 310
    camera_intrinsics.fx = 362.48
    camera_intrinsics.fy = 278.62

    def process_frame(frame_number):
        depth_frame_number = 0
        txt_file_path = os.path.join(args.marker_folder, f"{frame_number:06d}.txt")
        image_file_path = os.path.join(args.image_folder, f"{frame_number:06d}.png")
        depth_file_path = os.path.join(args.depth_folder, f"{depth_frame_number :06d}.{args.depth_format}")
        lidar_file_path = os.path.join(args.lidar_folder, f"{frame_number:06d}.pcd")
        front_image_path = os.path.join(args.front_image, f"{frame_number:06d}.png")
        
        if not os.path.exists(txt_file_path):
            print(f"Missing file: {txt_file_path}")
            return
        if not os.path.exists(image_file_path):
            print(f"Missing file: {image_file_path}")
            return
        if not os.path.exists(depth_file_path):
            print(f"Missing file: {depth_file_path}")
            return
        if not os.path.exists(lidar_file_path):
            print(f"Missing file: {lidar_file_path}")
            return
        if not os.path.exists(front_image_path):
            print(f"Missing file: {front_image_path}")
            return

        image = cv2.imread(image_file_path)

        if args.depth_format == 'npy':
            depth_data = np.load(depth_file_path)
            actual_depth_map = depth_data[0, 0]
            depth_max = np.max(actual_depth_map)
            normalized_depth = actual_depth_map / depth_max
            resized_depth_map = zoom(normalized_depth, (720 / normalized_depth.shape[0], 1280 / normalized_depth.shape[1]), order=1)
            reversed_depth_map = 1.0 - resized_depth_map
        elif args.depth_format == 'png':
            depth_data = cv2.imread(depth_file_path, cv2.IMREAD_UNCHANGED)
            reversed_depth_map = depth_data.astype(np.float32) / 1000

        markers = read_marker_info(txt_file_path)
        
        marker_positions_3d = []
        for x_norm, y_norm, radius in markers:
            x_pixel = int(x_norm * 1280)
            y_pixel = int(y_norm * 720)
            depth_raw = reversed_depth_map[y_pixel, x_pixel]
            
            if depth_raw < 0.1:
                depth_raw = calculate_depth_adjustment(reversed_depth_map, markers[0], markers[1], radius)
                    
            metric_coordinates = convert_depth_pixel_to_metric_coordinate(depth_intrinsics, [x_pixel, y_pixel], depth_raw)
            x_real, y_real, z_real = metric_coordinates

            print(f"Real-world coordinates at ({x_pixel}, {y_pixel}): ({x_real}, {y_real}, {z_real}) meters")
            marker_positions_3d.append([x_real, y_real, z_real])
            if args.visualize:
                image = draw_depth_vector(image, (x_pixel, y_pixel), depth_raw, scale=100)
                
            if len(marker_positions_3d) != len(set(map(tuple, marker_positions_3d))):
                print(f"Frame {frame_number}: Overlapping marker positions detected. Setting values to 0.")
                results.append([
                    frame_number,
                    0, 0, 0,  # headtracker_gaze_point_3d
                    0, 0, 0,  # roll, pitch, yaw
                    0, 0, 0,  # gaze_point_3d (KITTI)
                    0, 0, 0,
                    0, 0      # gaze_point_2d
                ])
                return

        centroid_x = np.mean([marker_positions_3d[0][0], marker_positions_3d[1][0]])
        centroid_y = np.mean([marker_positions_3d[0][1], marker_positions_3d[1][1]])
        centroid_z = np.mean([marker_positions_3d[0][2], marker_positions_3d[1][2]])

        # Calculate the depth difference and apply the ratio for accurate z
        z_diff = marker_positions_3d[2][2]
        
        
        normal_vector = calculate_plane_normal(np.array(marker_positions_3d[0]), np.array(marker_positions_3d[1]), np.array(marker_positions_3d[2]))
        theta = np.arctan2(normal_vector[1], normal_vector[2])
        accurate_z = z_diff + (0.24) * theta

        original_eyetracker_point = np.array([centroid_x, centroid_y, centroid_z])

        if len(marker_positions_3d) >= 3:
            p1, p2, p3 = np.array(marker_positions_3d[:3])
            normal_vector = calculate_plane_normal(p1, p2, p3)
            if normal_vector is not None:
                centroid_3d = np.mean([p1, p2, p3], axis=0)
                centroid_2d = (int(centroid_3d[0] * depth_intrinsics.fx / centroid_3d[2] + depth_intrinsics.ppx),
                            int(centroid_3d[1] * depth_intrinsics.fy / centroid_3d[2] + depth_intrinsics.ppy))

                image = draw_normal_vector(image, centroid_2d, normal_vector[:2], length=100)

                rotation_matrix = create_rotation_matrix(normal_vector)
                translation_vector = original_eyetracker_point
                transformation_matrix = create_transformation_matrix(rotation_matrix, translation_vector)

                gaze_row = gaze_data.iloc[depth_frame_number]
                gaze_point = np.array([gaze_row['.gaze_point_3d.x'], gaze_row['.gaze_point_3d.y'], gaze_row['.gaze_point_3d.z']]) / 100.0  # Convert to meters

                # 1. Eyetracker 좌표계에서 RealSense 좌표계로 변환
                gaze_point_rs = eyetracker_to_realsense(gaze_point)

                # 2. 회전 행렬 적용 및 평행 이동
                rotated_gaze_point = np.dot(rotation_matrix, gaze_point_rs)
                transformed_gaze_point_realsense = rotated_gaze_point + original_eyetracker_point

                # RealSense 기준의 gaze point
                headtracker_gaze_point_rs = transformed_gaze_point_realsense
                print(f"Headtracker gaze point (RealSense): {headtracker_gaze_point_rs}")

                # RealSense 기준의 roll, pitch, yaw 계산
                roll = 0  # Roll은 좌우 기울기이므로 일반적으로 수평 상태에서 0
                pitch = np.arctan2(-headtracker_gaze_point_rs[1], headtracker_gaze_point_rs[2])
                yaw = np.arctan2(headtracker_gaze_point_rs[0], headtracker_gaze_point_rs[2])

                transformed_gaze_point_2d = (int(transformed_gaze_point_realsense[0] * depth_intrinsics.fx / transformed_gaze_point_realsense[2] + depth_intrinsics.ppx),
                                            int(transformed_gaze_point_realsense[1] * depth_intrinsics.fy / transformed_gaze_point_realsense[2] + depth_intrinsics.ppy))
                cv2.circle(image, transformed_gaze_point_2d, 30, (255, 0, 0), -1)

                # eyetracker_point 계산
                eye_center_left = np.array([gaze_row['.eye_centers_3d_left.x'], gaze_row['.eye_centers_3d_left.y'], gaze_row['.eye_centers_3d_left.z']]) / 1000.0
                eye_center_right = np.array([gaze_row['.eye_centers_3d_right.x'], gaze_row['.eye_centers_3d_right.y'], gaze_row['.eye_centers_3d_right.z']]) / 1000.0
                eyetracker_point = (eye_center_left + eye_center_right) / 2

                # 1. Eyetracker 좌표계에서 RealSense 좌표계로 변환
                eyetracker_point_rs = eyetracker_to_realsense(eyetracker_point)

                # 2. 회전 행렬 적용 및 평행 이동
                rotated_eyetracker_point = np.dot(rotation_matrix, eyetracker_point_rs)
                transformed_eyetracker_point_realsense = rotated_eyetracker_point + original_eyetracker_point

                # RealSense 기준의 eyetracker point
                original_eyetracker_point = transformed_eyetracker_point_realsense
                eyetracker_point_rs = original_eyetracker_point
                print(f"Headtracker eyetracker point (RealSense): {original_eyetracker_point}")

                # RealSense 기준의 roll, pitch, yaw 계산
                roll = 0  # Roll은 좌우 기울기이므로 일반적으로 수평 상태에서 0
                pitch = np.arctan2(-headtracker_gaze_point_rs[1], headtracker_gaze_point_rs[2])
                yaw = np.arctan2(headtracker_gaze_point_rs[0], headtracker_gaze_point_rs[2])

            else:
                print(f"Frame {frame_number}: Unable to calculate normal vector (points may be collinear).")
        else:
            print(f"Frame {frame_number}: Insufficient marker data for plane calculation.")


        lidar_translation = np.array([0.4, 0.7, -1.2]) 
        lidar_transformation_matrix = create_transformation_matrix(np.eye(3), lidar_translation)

        lidar_gaze_point = apply_transformation(transformed_gaze_point_realsense, lidar_transformation_matrix)
        print(f"Lidar gaze point: {lidar_gaze_point}")

        # Transform the gaze point to KITTI coordinate system 
        kitti_gaze_point = realsense_to_lidar(lidar_gaze_point)  
        print(f"KITTI gaze point: {kitti_gaze_point}") 
        
        # Calculate the direction vector from headtracker to eyetracker in RealSense coordinate system
        gaze_direction_rs = transformed_gaze_point_realsense - original_eyetracker_point
        
        # Normalize the direction vector in RealSense coordinate system
        gaze_direction_rs_normalized = gaze_direction_rs / np.linalg.norm(gaze_direction_rs)
        
        # Calculate the translation from the LiDAR origin to the original eyetracker point
        eyetracker_lidar_translation = apply_transformation(original_eyetracker_point, lidar_transformation_matrix)
        eyetracker_lidar_gaze_point_rs = realsense_to_lidar(eyetracker_lidar_translation)
        print(f"Eyetracker LiDAR gaze point: {eyetracker_lidar_gaze_point_rs}")
        
        # Calculate the direction vector from headtracker to eyetracker in LiDAR coordinate system
        gaze_direction_lid =  headtracker_gaze_point_rs - eyetracker_lidar_gaze_point_rs

        # Normalize the direction vector in lidar coordinate system
        gaze_direction_lid_normalized = gaze_direction_lid / np.linalg.norm(gaze_direction_lid)
                
        results.append([
            frame_number,
            gaze_direction_rs_normalized[0], gaze_direction_rs_normalized[1], gaze_direction_rs_normalized[2],
            eyetracker_point_rs[0], eyetracker_point_rs[1], eyetracker_point_rs[2],
            gaze_direction_lid_normalized[0], gaze_direction_lid_normalized[1], gaze_direction_lid_normalized[2],
            eyetracker_lidar_gaze_point_rs[0], eyetracker_lidar_gaze_point_rs[1], eyetracker_lidar_gaze_point_rs[2],
            transformed_gaze_point_2d[0], transformed_gaze_point_2d[1]
        ])
        
        if args.visualize_3d:
            pcd = o3d.io.read_point_cloud(lidar_file_path) 
            
            # Set the point cloud to white color
            pcd.paint_uniform_color([1, 1, 1])
            
            lidar_translation = np.array([-0.4, 0.7, -1.2])
            lidar_zero = np.array([-0.4, 0.7, -0.7])
            
            # Create a line set to represent the gaze line
            extended_gaze_point = kitti_gaze_point * 5  # Extend the line 
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector([eyetracker_lidar_gaze_point_rs, extended_gaze_point])
            line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
            line_set.colors = o3d.utility.Vector3dVector([[0, 0, 1]])  # Blue color for gaze line

            # Create coordinate axes at the origin
            axis_length = 0.5  # Length of the axes
            origin = [0, 0, 0]
            x_axis = [axis_length, 0, 0]
            y_axis = [0, axis_length, 0]
            z_axis = [0, 0, axis_length]
            
            axis_points = [origin, x_axis, origin, y_axis, origin, z_axis]
            axis_lines = [[0, 1], [2, 3], [4, 5]]
            axis_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # Red, Green, Blue colors for x, y, z axes respectively

            axis_line_set = o3d.geometry.LineSet()
            axis_line_set.points = o3d.utility.Vector3dVector(axis_points)
            axis_line_set.lines = o3d.utility.Vector2iVector(axis_lines)
            axis_line_set.colors = o3d.utility.Vector3dVector(axis_colors)

            # Visualize the point cloud, gaze line, and coordinate axes
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            render_option = vis.get_render_option()
            render_option.background_color = np.asarray([0, 0, 0])  # Set background to black
            render_option.point_size = 1.0  # Set the point size to half the default for the point cloud
            vis.add_geometry(pcd)
            vis.add_geometry(line_set)
            vis.add_geometry(axis_line_set)
            vis.run()
            vis.destroy_window()

        # 프로젝트된 gaze point를 2D 이미지에 그리기
        if args.front_image:
            image_2d = cv2.imread(front_image_path)
            if image_2d is None:
                print(f"Error: Unable to load image from {front_image_path}")
                return

            # LiDAR 좌표계의 gaze point를 카메라 좌표계로 변환
            gaze_point_camera = lidar_to_camera(kitti_gaze_point, camera_extrinsics)

            print("Gaze point in camera coordinates:", gaze_point_camera)

            # y축 값이 0보다 작으면 음수로 설정하여 카메라 뒤쪽으로 투영되지 않도록 조정
            if gaze_point_camera[2] <= 0:
                print(f"Warning: Gaze point {gaze_point_camera} is behind the camera.")
                return

            # 이미지의 중심을 기준으로 좌표 계산
            image_x = int(camera_intrinsics.fx * gaze_point_camera[0] / gaze_point_camera[2] + camera_intrinsics.ppx)
            image_y = int(camera_intrinsics.fy * gaze_point_camera[1] / gaze_point_camera[2] + camera_intrinsics.ppy)

            print("Projected gaze point on image:", (image_x, image_y))

            # 2D 이미지에 gaze point 그리기
            if 0 <= image_x < image_2d.shape[1] and 0 <= image_y < image_2d.shape[0]:
                cv2.circle(image_2d, (image_x, image_y), 10, (0, 0, 255), -1)
            else:
                print(f"Warning: Gaze point {image_x, image_y} is outside the image bounds.")

            #output_image_path = os.path.join(args.front_image, f"gaze_projected_{frame_number:06d}.png")
            #cv2.imwrite(output_image_path, image_2d)

        depth_frame_number += 1

    if args.frame is not None:
        process_frame(args.frame)
    else:
        while True:
            process_frame(frame_number)
            frame_number += 1
            # frame이 존재하지 않을 때 break
            if not os.path.exists(os.path.join(args.marker_folder, f"{frame_number:06d}.txt")):
                print(f"No more files to process. Processed {frame_number} frames.")
                break

    # Save results to CSV
    columns = [
        'frame_number',
        '.H.gaze_normals_3d.x', '.H.gaze_normals_3d.y', '.H.gaze_normals_3d.z',
        '.H.eye_centers_3d.x', '.H.eye_centers_3d.y', '.H.eye_centers_3d.z',
        '.L.gaze_normals_3d.x', '.L.gaze_normals_3d.y', '.L.gaze_normals_3d.z',
        '.L.eye_centers_3d.x', '.L.eye_centers_3d.y', '.L.eye_centers_3d.z',
        '.gaze_point_2d.x', '.gaze_point_2d.y'
    ]
    results_df = pd.DataFrame(results, columns=columns)
    output_csv_path = os.path.join(os.path.dirname(os.path.dirname(args.image_folder)), 'Estimated_Gaze_point.csv')
    results_df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")

if __name__ == "__main__":
    main()
