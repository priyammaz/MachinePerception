'''
Question 5. Triangulation
In this question we move to 3D.
You are given keypoint matching between two images, together with the camera intrinsic and extrinsic matrix.
Your task is to perform triangulation to restore the 3D coordinates of the key points.
In your PDF, please visualize the 3d points and camera poses in 3D from three different viewing perspectives.
'''
import os
import cv2 # our tested version is 4.5.5
import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt
import random

# Coords of matched keypoint pairs in image 1 and 2, dims (#matches, 4). Same pair of images as before
# For each row, it consists (k1_x, k1_y, k2_x, k2_y).
# If necessary, you can convert float to int to get the integer coordinate
all_good_matches = np.load('assets/all_good_matches.npy')

K1 = np.load('assets/fountain/Ks/0000.npy')
K2 = np.load('assets/fountain/Ks/0005.npy')

R1 = np.load('assets/fountain/Rs/0000.npy')
R2 = np.load('assets/fountain/Rs/0005.npy')

t1 = np.load('assets/fountain/ts/0000.npy')
t2 = np.load('assets/fountain/ts/0005.npy')

def triangulate(K1, K2, R1, R2, t1, t2, all_good_matches):
    """
    Arguments:
        K1: intrinsic matrix for image 1, dim: (3, 3)
        K2: intrinsic matrix for image 2, dim: (3, 3)
        R1: rotation matrix for image 1, dim: (3, 3)
        R2: rotation matrix for image 1, dim: (3, 3)
        t1: translation for image 1, dim: (3,)
        t2: translation for image 1, dim: (3,)
        all_good_matches:  dim: (#matches, 4)
    Returns:
        points_3d, dim: (#matches, 3)
    """
    points_3d = None
    # Create Projection Matricies
    p1 = np.dot(K1, np.hstack([R1, t1]))
    p2 = np.dot(K2, np.hstack([R2, t2]))

    img1 = np.hstack([all_good_matches[:, :2], np.ones(shape=(len(all_good_matches),1))]).T
    img2 = np.hstack([all_good_matches[:, 2:], np.ones(shape=(len(all_good_matches),1))]).T

    point_3d = []
    for i in range(len(all_good_matches)):
        a = np.array([[img1[1, i] * p1[2, :] - p1[1, :]],
                      [p1[0, :] - img1[0,i]*p1[2, :]],
                      [img2[1, i] * p2[2, :] - p2[1, :]],
                      [p2[0, :] - img2[0, i] * p2[2, :]]]).reshape(4,4)

        _, _, V = np.linalg.svd(a)
        point = V[-1]
        point = point/point[-1]
        point_3d.append(point)

    point_3d = np.array(point_3d)[:, :3]

    return point_3d

points_3d = triangulate(K1, K2, R1, R2, t1, t2, all_good_matches)

if points_3d is not None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    
    # Visualize both point and camera
    # Check this link for Open3D visualizer http://www.open3d.org/docs/release/tutorial/visualization/visualization.html#Function-draw_geometries
    # Check this function for adding a virtual camera in the visualizer http://www.open3d.org/docs/release/tutorial/visualization/visualization.html#Function-draw_geometries
    # Open3D is not the only option. You could use matplotlib, vtk or other visualization tools as well.
    # --------------------------- Begin your code here ---------------------------------------------
    extrinsic_1 = np.vstack([np.hstack([R1, t1]), [0,0,0,1]])
    extrinsic_2 = np.vstack([np.hstack([R2, t2]), [0, 0, 0, 1]])
    virtual_cam_1 = o3d.geometry.LineSet.create_camera_visualization(640, 320, K1, extrinsic_1)
    virtual_cam_2 = o3d.geometry.LineSet.create_camera_visualization(640, 320, K2, extrinsic_2)
    o3d.visualization.draw_geometries([virtual_cam_1, virtual_cam_2, pcd])

    # --------------------------- End your code here   ---------------------------------------------