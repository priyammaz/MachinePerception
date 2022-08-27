from re import I
import time
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import time
from sklearn.neighbors import KDTree

# Question 4: deal with point_to_plane = True
def fit_rigid(src, tgt, point_to_plane = False):
  # Question 2: Rigid Transform Fitting
  # Implement this function
  # -------------------------
  if point_to_plane is False:
    T = np.identity(4)
    # Center points
    src_n = src - src.mean(axis=0)
    tgt_n = tgt - tgt.mean(axis=0)

    # Calculate rotation matrix
    U, S, V = np.linalg.svd(np.dot(src_n.T, tgt_n))
    R = np.dot(V.T, U.T)

    # Calculate Translation Matrix
    t = tgt.mean(axis=0) - np.dot(R, src.mean(axis=0))

    T[:3, :3] = R
    T[:3, 3] = t
    
  else:
    T = np.identity(4)
    sum_matrix = np.zeros(shape=(6,6))
    b_matrix = np.zeros(shape=(6,1))
    normals = np.array(source.normals)
    for i in range(len(src)):
      src_pt = src[i]
      tgt_pt = tgt[i]
      n = normals[i]
      c = np.cross(src_pt, n)
      mat = [[c[0]*c[0], c[0]*c[1], c[0]*c[2], c[0]*n[0], c[0]*n[1], c[0]*n[2]],
             [c[1]*c[0], c[1]*c[1], c[1]*c[2], c[1]*n[0], c[1]*n[1], c[1]*n[2]],
             [c[2]*c[0], c[2]*c[1], c[2]*c[2], c[2]*n[0], c[2]*n[1], c[2]*n[2]],
             [n[0]*c[0], n[0]*c[1], n[0]*c[2], n[0]*n[0], n[0]*n[1], n[0]*n[2]],
             [n[1]*c[0], n[1]*c[1], n[1]*c[2], n[1]*n[0], n[1]*n[1], n[1]*n[2]],
             [n[2]*c[0], n[2]*c[1], n[2]*c[2], n[2]*n[0], n[2]*n[1], n[2]*n[2]]]

      mat = np.array(mat)

      b = np.array([np.dot(c[0]*(src_pt - tgt_pt), n),
                    np.dot(c[1]*(src_pt - tgt_pt), n),
                    np.dot(c[2]*(src_pt - tgt_pt), n),
                    np.dot(n[0]*(src_pt - tgt_pt), n),
                    np.dot(n[1]*(src_pt - tgt_pt), n),
                    np.dot(n[2]*(src_pt - tgt_pt), n)])
      b = b.reshape((6,1))
      sum_matrix = sum_matrix + mat
      b_matrix = b_matrix - b

    sol = np.linalg.solve(a=sum_matrix, b=b_matrix).flatten()

    R = np.array([[1, -sol[2], sol[1]],
                  [sol[2], 1, -sol[0]],
                  [-sol[1], sol[0], 1]])
    t = sol[3:]
    T[:3, :3] = R
    T[:3, 3] = t
  return T

# Question 4: deal with point_to_plane = True
def icp(source, target, init_pose=np.eye(4), max_iter = 20, point_to_plane = False):
  try:
    src = np.asarray(source.points)
    tgt = np.asarray(target.points)
  except:
    src = source
    tgt = target

  # Question 3: ICP
  # Hint 1: using KDTree for fast nearest neighbour
  # Hint 3: you should be calling fit_rigid inside the loop
  # You implementation between the lines
  # ---------------------------------------------------
  T = init_pose
  transforms = []
  delta_Ts = []
  tree = KDTree(tgt)
  threshold = 0.02
  # Create homogenous points
  src = np.hstack((src, np.ones(len(src)).reshape(-1, 1)))
  tgt = np.hstack((tgt, np.ones(len(tgt)).reshape(-1, 1)))

  for i in range(max_iter):
    T_delta = np.identity(4)
    # Get distance and index of nearest neighbors
    dist, ind = tree.query(src[:, :3], k=1)
    inlier_ratio = (dist < threshold).sum()/len(dist)
    T_delta = fit_rigid(src[:, :3], tgt[ind, :3].reshape(-1, 3), point_to_plane=point_to_plane)

    # src = np.dot(T_delta, src.T).T
    src = (T_delta @ src.T).T

    if len(transforms) != 0:
       T = T_delta @ transforms[-1]
    else:
      T = T_delta

    if inlier_ratio > 0.999:
      break

    print("iter %d: inlier ratio: %.2f" % (i+1, inlier_ratio))
    # relative update from each iteration
    delta_Ts.append(T_delta.copy())
    # pose estimation after each iteration
    transforms.append(T.copy())
  return transforms, delta_Ts

def rgbd2pts(color_im, depth_im, K):
  # Question 1: unproject rgbd to color point cloud, provide visualiation in your document
  # Your implementation between the lines

  # plt.imshow(color_im)
  # plt.show()
  # Generate 3D Coordinates
  x = np.linspace(0, color_im.shape[1] - 1, color_im.shape[1]).astype(np.int32)
  y = np.linspace(0, color_im.shape[0] - 1, color_im.shape[0]).astype(np.int32)
  xs, ys = np.meshgrid(x, y)
  stack = np.vstack([xs.flatten(), ys.flatten()])
  ones_list = np.ones(shape=(1, stack.shape[1]))
  stack = np.vstack([stack, ones_list])

  k_inv = np.linalg.inv(K)
  xyz = np.multiply(np.dot(k_inv, stack), depth_im.flatten())
  color = np.moveaxis(color_im, 2, 0).reshape(3, -1)
  
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(xyz.T)
  pcd.colors = o3d.utility.Vector3dVector(color.T)

  # o3d.visualization.draw_geometries([pcd])
  return pcd

# TODO (Shenlong): please check that I set this question up correctly, it is called on line 136
def pose_error(estimated_pose, gt_pose):
  # Question 5: Translation and Rotation Error 
  # Use equations 5-6 in https://cmp.felk.cvut.cz/~hodanto2/data/hodan2016evaluation.pdf
  # Your implementation between the lines
  # ---------------------------
  error = 0
  estimated_r = estimated_pose[:3, :3]
  gt_r = gt_pose[:3, :3]
  estimated_t = estimated_pose[:3, 3]
  gt_t = gt_pose[:3, 3]

  translation_error =  np.linalg.norm(estimated_t - gt_t)
  rotation_error = np.arccos((np.trace(estimated_r @ np.linalg.inv(gt_r)) - 1) / 2)

  error = (rotation_error, translation_error)
  # ---------------------------
  return error

def read_data(ind = 0):
  K = np.loadtxt("data/camera-intrinsics.txt", delimiter=' ')
  depth_im = cv2.imread("data/frame-%06d.depth.png"%(ind),-1).astype(float)
  depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
  depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
  T = np.loadtxt("data/frame-%06d.pose.txt"%(ind))  # 4x4 rigid transformation matrix
  color_im = cv2.imread("data/frame-%06d.color.jpg"%(ind),-1)
  color_im = cv2.cvtColor(color_im, cv2.COLOR_BGR2RGB)  / 255.0
  return color_im, depth_im, K, T

if __name__ == "__main__":

  # pairwise ICP

  # read color, image data and the ground-truth, converting to point cloud
  color_im, depth_im, K, T_tgt = read_data(0)
  target = rgbd2pts(color_im, depth_im, K)
  color_im, depth_im, K, T_src = read_data(40)
  source = rgbd2pts(color_im, depth_im, K)

  # downsampling and normal estimatoin
  source = source.voxel_down_sample(voxel_size=0.02)
  target = target.voxel_down_sample(voxel_size=0.02)
  source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
  target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

  # # # conduct ICP (your code)
  final_Ts, delta_Ts = icp(source, target, point_to_plane=False)

  # visualization
  vis = o3d.visualization.Visualizer()
  vis.create_window()
  ctr = vis.get_view_control()
  ctr.set_front([ -0.11651295252277051, -0.047982289143896774, -0.99202945108647766 ])
  ctr.set_lookat([ 0.023592929264511786, 0.051808635289583765, 1.7903649529102956 ])
  ctr.set_up([ 0.097655832648056065, -0.9860023571949631, -0.13513952033284915 ])
  ctr.set_zoom(0.42199999999999971)
  vis.add_geometry(source)
  vis.add_geometry(target)

  save_image = False

  # update source images
  for i in range(len(delta_Ts)):
      source.transform(delta_Ts[i])
      vis.update_geometry(source)
      vis.poll_events()
      vis.update_renderer()
      time.sleep(0.2)
      if save_image:
          vis.capture_screen_image("temp_%04d.jpg" % i)

  # visualize camera
  h, w, c = color_im.shape
  tgt_cam = o3d.geometry.LineSet.create_camera_visualization(w, h, K, np.eye(4), scale = 0.2)
  src_cam = o3d.geometry.LineSet.create_camera_visualization(w, h, K, np.linalg.inv(T_src) @ T_tgt, scale = 0.2)
  pred_cam = o3d.geometry.LineSet.create_camera_visualization(w, h, K, np.linalg.inv(final_Ts[-1]), scale = 0.2)

  gt_pose = np.linalg.inv(T_src) @ T_tgt
  pred_pose = np.linalg.inv(final_Ts[-1])
  p_error = pose_error(pred_pose, gt_pose)
  print("Ground truth pose:", gt_pose)
  print("Estimated pose:", pred_pose)
  print("Rotation/Translation Error", p_error)

  tgt_cam.paint_uniform_color((1, 0, 0))
  src_cam.paint_uniform_color((0, 1, 0))
  pred_cam.paint_uniform_color((0, 0.5, 0.5))
  vis.add_geometry(src_cam)
  vis.add_geometry(tgt_cam)
  vis.add_geometry(pred_cam)

  vis.run()
  vis.destroy_window()

  # Provide visualization of alignment with camera poses in write-up.
  # Print pred pose vs gt pose in write-up.