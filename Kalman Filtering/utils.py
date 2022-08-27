import numpy as np
from kitti_oxts import roty
import warnings

def load_detection(file):
    # load from raw file
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dets = np.loadtxt(file, delimiter=',')     # load detections, N x 15

    if len(dets.shape) == 1: dets = np.expand_dims(dets, axis=0)     
    if dets.shape[1] == 0:        # if no detection in a sequence
        return [], False
    else:
        return dets, True

def process_dets(dets):
    # convert each detection into the class Box3D 
    # inputs: 
    #     dets - a numpy array of detections in the format [[h,w,l,x,y,z,theta],...]

    dets_new = []
    for det in dets:
        det_tmp = Box3D.array2bbox_raw(det)
        dets_new.append(det_tmp)

    return dets_new

def get_frame_det(dets_all, frame):
    # get irrelevant information associated with an object, not used for associationg
    ori_array = dets_all[dets_all[:, 0] == frame, -1].reshape((-1, 1))        # orientation
    other_array = dets_all[dets_all[:, 0] == frame, 1:7]                     # other information, e.g, 2D box, ...
    additional_info = np.concatenate((ori_array, other_array), axis=1)        

    # get 3D box
    dets = dets_all[dets_all[:, 0] == frame, 7:14]        

    dets_frame = {'dets': dets, 'info': additional_info}
    return dets_frame

def within_range(theta):
    # make sure the orientation is within a proper range

    if theta >= np.pi: theta -= np.pi * 2    # make the theta still in the range
    if theta < -np.pi: theta += np.pi * 2

    return theta

def orientation_correction(theta_pre, theta_obs):
        # update orientation in propagated tracks and detected boxes so that they are within 90 degree
        
        # make the theta still in the range
        theta_pre = within_range(theta_pre)
        theta_obs = within_range(theta_obs)

        # if the angle of two theta is not acute angle, then make it acute
        if abs(theta_obs - theta_pre) > np.pi / 2.0 and abs(theta_obs - theta_pre) < np.pi * 3 / 2.0:     
            theta_pre += np.pi       
            theta_pre = within_range(theta_pre)

        # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
        if abs(theta_obs - theta_pre) >= np.pi * 3 / 2.0:
            if theta_obs > 0: theta_pre += np.pi * 2
            else: theta_pre -= np.pi * 2

        return theta_pre, theta_obs

class Box3D:
    def __init__(self, x=None, y=None, z=None, h=None, w=None, l=None, ry=None, s=None):
        self.x = x      # center x
        self.y = y      # center y
        self.z = z      # center z
        self.h = h      # height
        self.w = w      # width
        self.l = l      # length
        self.ry = ry    # orientation
        self.s = s      # detection score
        self.corners_3d_cam = None

    def __str__(self):
        return 'x: {}, y: {}, z: {}, heading: {}, length: {}, width: {}, height: {}, score: {}'.format(
            self.x, self.y, self.z, self.ry, self.l, self.w, self.h, self.s)
    
    @classmethod
    def bbox2dict(cls, bbox):
        return {
            'center_x': bbox.x, 'center_y': bbox.y, 'center_z': bbox.z,
            'height': bbox.h, 'width': bbox.w, 'length': bbox.l, 'heading': bbox.ry}
    
    @classmethod
    def bbox2array(cls, bbox):
        if bbox.s is None:
            return np.array([bbox.x, bbox.y, bbox.z, bbox.ry, bbox.l, bbox.w, bbox.h])
        else:
            return np.array([bbox.x, bbox.y, bbox.z, bbox.ry, bbox.l, bbox.w, bbox.h, bbox.s])

    @classmethod
    def bbox2array_raw(cls, bbox):
        if bbox.s is None:
            return np.array([bbox.h, bbox.w, bbox.l, bbox.x, bbox.y, bbox.z, bbox.ry])
        else:
            return np.array([bbox.h, bbox.w, bbox.l, bbox.x, bbox.y, bbox.z, bbox.ry, bbox.s])

    @classmethod
    def array2bbox_raw(cls, data):
        # take the format of data of [h,w,l,x,y,z,theta]

        bbox = Box3D()
        bbox.h, bbox.w, bbox.l, bbox.x, bbox.y, bbox.z, bbox.ry = data[:7]
        if len(data) == 8:
            bbox.s = data[-1]
        return bbox
    
    @classmethod
    def array2bbox(cls, data):
        # take the format of data of [x,y,z,theta,l,w,h]

        bbox = Box3D()
        bbox.x, bbox.y, bbox.z, bbox.ry, bbox.l, bbox.w, bbox.h = data[:7]
        if len(data) == 8:
            bbox.s = data[-1]
        return bbox
    
    @classmethod
    def box2corners3d_camcoord(cls, bbox):
        ''' Takes an object's 3D box with the representation of [x,y,z,theta,l,w,h] and 
            convert it to the 8 corners of the 3D box, the box is in the camera coordinate
            with right x, down y, front z
            
            Returns:
                corners_3d: (8,3) array in in rect camera coord

            box corner order is like follows
                    1 -------- 0         top is bottom because y direction is negative
                   /|         /|
                  2 -------- 3 .
                  | |        | |
                  . 5 -------- 4
                  |/         |/
                  6 -------- 7    
            
            rect/ref camera coord:
            right x, down y, front z

            x -> w, z -> l, y -> h
        '''

        # if already computed before, then skip it
        if bbox.corners_3d_cam is not None:
            return bbox.corners_3d_cam

        # compute rotational matrix around yaw axis
        # -1.57 means straight, so there is a rotation here
        R = roty(bbox.ry)   

        # 3d bounding box dimensions
        l, w, h = bbox.l, bbox.w, bbox.h

        # 3d bounding box corners
        x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
        y_corners = [0,0,0,0,-h,-h,-h,-h]
        z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]

        # rotate and translate 3d bounding box
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0,:] = corners_3d[0,:] + bbox.x
        corners_3d[1,:] = corners_3d[1,:] + bbox.y
        corners_3d[2,:] = corners_3d[2,:] + bbox.z
        corners_3d = np.transpose(corners_3d)
        bbox.corners_3d_cam = corners_3d

        return corners_3d