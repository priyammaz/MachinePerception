import numpy as np, cv2, random, copy
from PIL import Image
from utils import Box3D

random.seed(0)

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    import colorsys
    brightness = 1.0 if bright else 0.7
    hsv = [(i / float(N), 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    # random.shuffle(colors)
    return colors

max_color = 30
colors = random_colors(max_color)       # Generate random colors

def draw_box3d_image(image, qs, img_size=(900, 1600), color=(255,255,255), thickness=4):
    ''' Draw 3d bounding box in image
        qs: (8,2) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''

    def check_outside_image(x, y, height, width):
        if x < 0 or x >= width: return True
        if y < 0 or y >= height: return True

    # if 6 points of the box are outside the image, then do not draw
    pts_outside = 0
    for index in range(8):
        check = check_outside_image(qs[index, 0], qs[index, 1], img_size[0], img_size[1])
        if check: pts_outside += 1
    if pts_outside >= 6: return image, False

    # actually draw
    if qs is not None:
        qs = qs.astype(np.int32)
        for k in range(0,4):
            i,j=k,(k+1)%4
            image = cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA) # use LINE_AA for opencv3

            i,j=k+4,(k+1)%4 + 4
            image = cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

            i,j=k,k+4
            image = cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

    return image, True

def vis_obj(box, img, calib, hw, color_tmp=None, str_vis=None, thickness=4, id_hl=None, err_type=None):
    # visualize an individual object    
    # repeat is for highlighted objects, used to create pause in the video

    # draw box
    obj_8corner = Box3D.box2corners3d_camcoord(box)
    obj_pts_2d = calib.project_rect_to_image(obj_8corner)
    img, draw = draw_box3d_image(img, obj_pts_2d, hw, color=color_tmp, thickness=thickness)

    # draw text
    if draw and obj_pts_2d is not None and str_vis is not None:
        x1, y1 = int(obj_pts_2d[4, 0]), int(obj_pts_2d[4, 1])
        img = cv2.putText(img, str_vis, (x1+5, y1-10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color_tmp, int(thickness/2))

    # highlight
    if err_type is not None:
        
        # compute the radius of the highlight
        xmin = np.min(obj_pts_2d[:, 0]); xmax = np.max(obj_pts_2d[:, 0])
        ymin = np.min(obj_pts_2d[:, 1]); ymax = np.max(obj_pts_2d[:, 1])
        radius = int(max(ymax - ymin, xmax - xmin) / 2 * 1.5)
        radius = max(radius, 50)

        # draw highlighting circle
        center = np.average(obj_pts_2d, axis=0)
        center = tuple(center.astype('int16'))
        img = cv2.circle(img, center, radius, (255, 0, 0), 4)        

        # draw error message
        pos_x, pos_y = center[0] - radius, center[1] - radius - 10
        font = cv2.FONT_HERSHEY_TRIPLEX
        font_scale = 1
        font_thickness = 2
        text_size, _ = cv2.getTextSize(err_type, font, font_scale, font_thickness)
        text_w, text_h = text_size
        cv2.rectangle(img, (pos_x, pos_y - text_h - 5), (pos_x + text_w, pos_y + 5), (255, 255, 255), -1)         # add white background
        img = cv2.putText(img, err_type, (pos_x, pos_y), font, font_scale, (255, 0, 0), font_thickness) 

    return img

def vis_image_with_obj(img, obj_res, obj_gt, calib, save_path, h_thres=0, \
    color_type='det', id_hl=None, repeat=60):
    # obj_res, obj_gt, a list of object3D class instances
    # h_thres: height threshold for filtering objects
    # id_hl: ID to be highlighted, color_type: ['det', 'trk'], trk means different color for each one
    # det means th same color for the same object
    hw = (375, 1242)

    # load image
    img = np.array(Image.open(img))

    # loop through every objects
    for obj in obj_res:
        depth = obj.x[2]
        if depth >= 2:         # check in front of camera

            # obtain object color and thickness
            if color_type == 'trk':   
                color_id = obj.ID         # vary across objects
                thickness = 5
            elif color_type == 'det': 
                if id_hl is not None and obj.ID in id_hl:
                    # color_id = 29         # fixed, red for highlighted errors
                    color_id = obj.ID * 9     # some magic number to scale up the id so that nearby ID does not have similar color
                    thickness = 5            # highlighted objects are thicker
                else:                        
                    color_id = 13            # general object, fixed, lightgreen
                    thickness = 1            # general objects are thin
            color_tmp = tuple([int(tmp * 255) for tmp in colors[color_id % max_color]])

            # get miscellaneous information
            # box_tmp = obj.get_box3D()
            box_tmp = Box3D.array2bbox(obj.x.reshape((-1))[:7])
            
            str_vis = 'ID: %d' % obj.ID
            
            # retrieve index in the id_hl dict
            if id_hl is not None and obj.ID in id_hl:
                err_type = id_hl[obj.ID]
            else:
                err_type = None
            img = vis_obj(box_tmp, img, calib, hw, color_tmp, str_vis, thickness, id_hl, err_type)

    # save image
    img = Image.fromarray(img)
    img = img.resize((hw[1], hw[0]))
    img.save(save_path)

    # create copy of the same image with highlighted objects to pause
    if id_hl is not None:
        for repeat_ in range(repeat):
            save_path_tmp = save_path[:-4] + '_repeat_%d' % repeat_ + save_path[-4:]
            img.save(save_path_tmp)

def visualization(img, dets, trks, trackers, calib, save_path):
    from PIL import Image
    
    dets, trks = copy.copy(dets), copy.copy(trks)
    # img = np.array(Image.open(img))
    max_color = 20
    colors = random_colors(max_color) # Generate random colors
    hw = (375, 1242) #(img.size[1], img.size[0])

    # visualize all detections as yellow boxes
    for det_tmp in dets: 
        img = vis_obj(det_tmp, img, calib, hw, (255, 0, 0))
    
    # visualize color-specific tracks
    count = 0
    ID_list = [tmp.ID for tmp in trackers]
    for trk_tmp in trks: 
        ID_tmp = ID_list[count]
        color_float = colors[int(ID_tmp) % max_color]
        color_int = tuple([int(tmp * 255) for tmp in color_float])
        str_vis = "{}".format(ID_tmp)
        # str_vis = None
        # img = vis_obj(trk_tmp, img, calib, hw, color_int, str_vis) 
        img = vis_obj(trk_tmp, img, calib, hw, (0,255,0), str_vis)# green for tracklets
        count += 1
    
    img = Image.fromarray(img)
    img = img.resize((hw[1], hw[0]))
    img.save(save_path)