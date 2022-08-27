import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

# read intrinsics, extrinsincs and camera images
K1 = np.load('assets/fountain/Ks/0005.npy')
K2 = np.load('assets/fountain/Ks/0004.npy')
R1 = np.load('assets/fountain/Rs/0005.npy')
R2 = np.load('assets/fountain/Rs/0004.npy')
t1 = np.load('assets/fountain/ts/0005.npy')
t2 = np.load('assets/fountain/ts/0004.npy')
img1 = cv2.imread('assets/fountain/images/0005.png')
img2 = cv2.imread('assets/fountain/images/0004.png')
h, w, _ = img1.shape

# resize the image to reduce computation
scale = 8 # you could try different scale parameters, e.g. 4 for better quality & slower speed.
img1 = cv2.resize(img1, (w//scale, h//scale))
img2 = cv2.resize(img2, (w//scale, h//scale))
h, w, _ = img1.shape

# visualize the left and right image
plt.figure()
# opencv default color order is BGR instead of RGB so we need to take care of it when visualization
plt.imshow(cv2.cvtColor(np.concatenate((img1, img2), axis=1), cv2.COLOR_BGR2RGB))
plt.title("Before rectification")
# plt.show()

# Q6.a: How does intrinsic change before and after the scaling?
# You only need to modify K1 and K2 here, if necessary. If you think they remain the same, leave here as blank and explain why.
K1[:2, :] = K1[:2, :] / scale
K2[:2, :] = K2[:2, :] / scale


# Compute the relative pose between two cameras
T1 = np.eye(4)
T1[:3, :3] = R1
T1[:3, 3:] = t1
T2 = np.eye(4)
T2[:3, :3] = R2
T2[:3, 3:] = t2
T = T2.dot(np.linalg.inv(T1)) # c1 to world and world to c2
R = T[:3, :3]
t = T[:3, 3:]

# Rectify stereo image pair such that they are frontal parallel. Here we call cv2 to help us
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(K1, None, K2, None,(w // 4, h // 4), R, t, 1, newImageSize=(0,0))
left_map  = cv2.initUndistortRectifyMap(K1, None, R1, P1, (w, h), cv2.CV_16SC2)
right_map = cv2.initUndistortRectifyMap(K2, None, R2, P2, (w, h), cv2.CV_16SC2)
left_img = cv2.remap(img1, left_map[0],left_map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
right_img = cv2.remap(img2, right_map[0],right_map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
plt.figure()
plt.imshow(cv2.cvtColor(np.concatenate((left_img, right_img), axis = 1), cv2.COLOR_BGR2RGB))
plt.title("After stereo rectification")
# plt.show()
# Visualize images after rectification and report K1, K2 in your PDF report.


def stereo_matching_ssd(left_im, right_im, max_disp = 128, block_size = 7):
  """
  Using sum of square difference to compute stereo matching.
  Arguments:
      left_im: left image (h x w x 3 numpy array)
      right_im: right image (h x w x 3 numpy array)
      mask_disp: maximum possible disparity
      block_size: size of the block for computing matching cost
  Returns:
      disp_im: disparity image (h x w numpy array), storing the disparity values
  """
  disp_map = np.zeros_like(left_im[:, :, 0])
  kernel_left = block_size//2
  kernel_right = block_size - kernel_left
  h, w, _ = left_im.shape

  for y in tqdm(range(kernel_left, h - kernel_left)):
      for x in range(kernel_left, w - kernel_left):
          kernel_y_min = y - kernel_left
          kernel_y_max = y + kernel_right
          kernel_x_min = x - kernel_left
          kernel_x_max = x + kernel_right
          left_section = left_im[kernel_y_min:kernel_y_max, kernel_x_min:kernel_x_max]
          off_range = min(kernel_x_min, max_disp)

          best_disparity = 0
          best_ssd = np.inf
          for offset in range(off_range+1):
              right_section = right_im[kernel_y_min:kernel_y_max, kernel_x_min - offset:kernel_x_max - offset]
              ssd = np.sum((left_section - right_section)**2)

              if ssd < best_ssd:
                  best_disparity = offset
                  best_ssd = ssd

          disp_map[y, x] = best_disparity
      
  return disp_map

disparity = stereo_matching_ssd(left_img, right_img, max_disp = 128, block_size=7)
# Depending on your implementation, runtime could be a few minutes.
# Feel free to try different hyper-parameters, e.g. using a higher-resolution image, or a bigger block size. Do you see any difference?

plt.figure()
plt.imshow(disparity)
plt.title("Disparity map")
plt.show()

# Compare your method and an off the shelf CV2's stereo matching results.
# Please list a few directions which you think could improve your own results
left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
plt.imshow(np.concatenate((left_gray, right_gray), axis = 1))
stereo = cv2.StereoBM_create(numDisparities=128, blockSize=7)
disparity_cv2 = stereo.compute(left_gray, right_gray) / 16.0
plt.imshow(np.concatenate((disparity, disparity_cv2), axis = 1))
plt.show()

# Visualize disparity map and comparison against disparity_cv2 in your report.


# Q6 Bonus:


# --------------------------- Begin your code here ---------------------------------------------
xyz = None
color = None
# --------------------------- End your code here   ---------------------------------------------

# Hints:
# What is the focal length? How large is the stereo baseline?
# Convert disparity to depth
# Unproject image color and depth map to 3D point cloud
# You can use Open3D to visualize the colored point cloud

if xyz is not None:
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(xyz)
  pcd.colors = o3d.utility.Vector3dVector(color)
  o3d.visualization.draw_geometries([pcd])