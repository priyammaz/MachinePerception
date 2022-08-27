import numpy as np
import matplotlib.pyplot as plt
from imageio import imread

# You could pip install the following dependencies if any is missing
# pip install -r requirements.txt

# Load the image and plot the keypoints
im = imread('uiuc.png') / 255.0
keypoints_im = np.array([(604.593078169188, 583.1361439828671),
                       (1715.3135416380655, 776.304920238324),
                       (1087.5150188078305, 1051.9034760165837),
                       (79.20731171576836, 642.2524505093215)])


'''
Question 1: specify the corners' coordinates
Take point 3 as origin, the long edge as x axis and short edge as y axis
Output:
     - corners_court: a numpy array (4x2 matrix)
'''
# --------------------------- Begin your code here ---------------------------------------------

corners_court = [[0, 15.24],
                 [28.65, 15.24],
                 [28.65, 0],
                 [0,0]]

corners_court = np.array(corners_court)
# --------------------------- End your code here   ---------------------------------------------

'''
Question 2: complete the findHomography function
Arguments:
     pts_src - Each row corresponds to an actual point on the 2D plane (Nx2 matrix)
     pts_dst - Each row is the pixel location in the target image coordinate (Nx2 matrix)
     back_image - Each row is the pixel location in the back image where Z=150
Returns:
     H - The homography matrix (3x3 matrix)

Hints:
    - you might find the function vstack, hstack to be handy for getting homogenous coordinate;
    - you might find numpy.linalg.svd to be useful for solving linear system
    - directly calling findHomography in cv2 will receive zero point, but you could use it as sanity-check of your own implementation
'''
def findHomography(pts_src, pts_dst):
    matrix = []
    for idx in range(len(pts_src)):
        cp_x, cp_y = pts_src[idx][0], pts_src[idx][1]
        kp_x, kp_y = pts_dst[idx][0], pts_dst[idx][1]
        matrix.append([0, 0, 0, cp_x, cp_y, 1, -1*kp_y*cp_x, -1*kp_y*cp_y, -1*kp_y])   # cp_x, cp_y, 1 converts to homogenous
        matrix.append([cp_x, cp_y, 1, 0, 0, 0, -1*kp_x*cp_x, -1*kp_x*cp_y, -1*kp_x])   # cp_x, cp_y, 1 converts to homogenous
    matrix = np.array(matrix) # Convert to a numpy array
    ### USE SVD TO SOLVE FOR LAST SINGULAR VECTOR ###
    U, S, V = np.linalg.svd(matrix)
    return V[-1].reshape(3,3)  # Return reshaped last singular vector

# Calculate the homography matrix using your implementation

H = findHomography(corners_court, keypoints_im)


'''
# Question 3.a: insert the logo virtually onto the state farm center image.
# Specific requirements:
#      - the size of the logo needs to be 3x6 meters;
#      - the bottom left logo corner is at the location (23, 2) on the basketball court.
# Returns:
#      transform_target - The transformation matrix from logo.png image coordinate to target.png coordinate (3x3 matrix)
#
# Hints:
#      - Consider to calculate the transform as the composition of the two: H_logo_target = H_logo_court @ H_court_target
#      - Given the banner size in meters and image size in pixels, could you scale the logo image coordinate from pixels to meters
#      - What transform will move the logo to the target location?
#      - Could you leverage the homography between basketball court to target image we computed in Q.2?
#      - Image coordinate is y down ((0, 0) at bottom-left corner) while we expect the inserted logo to be y up, how would you handle this?
# '''

# Read the banner image that we want to insert to the basketball court
logo = imread('logo.png') / 255.0
# --------------------------- Begin your code here ---------------------------------------------
banner_coor = np.array([[0,500], [1000,500], [1000,0], [0,0]])
box_coor = np.array([[23,5], [29,5], [29,2], [23,2]])
png_2_box_coor_hom = findHomography(banner_coor, box_coor)
target_transform = H @ png_2_box_coor_hom
# --------------------------- End your code here   ---------------------------------------------

'''
Question 3.b: compute the warpImage function
Arguments:
     image - the source image you may want to warp (Hs x Ws x 4 matrix, R,G,B,alpha)
     H - the homography transform from the source to the target image coordinate (3x3 matrix)
     shape - a tuple of the target image shape (Wt, Ht)
Returns:
     image_warped - the warped image (Ht x Wt x 4 matrix)

Hints:
    - you might find the function numpy.meshgrid and numpy.ravel_multi_index useful;
    - are you able to get rid of any for-loop over all pixels?
    - directly calling warpAffine or warpPerspective in cv2 will receive zero point, but you could use as sanity-check of your own implementation
'''

def warpImage(image, H, shape):
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    xs, ys = np.meshgrid(x, y)

    im = np.zeros(shape=(shape[1], shape[0], 4))

    coordinates = np.stack((xs, ys), axis=-1).reshape(len(x) * len(y), 2)
    homo_coordinates = np.hstack((coordinates, np.ones(shape=(len(coordinates), 1)))).T
    homo_coordinates[1, :] = image.shape[0] - homo_coordinates[1, :] # Flip y coordinate so we are in the correct orientation
    transformed_coordinates = H.dot(homo_coordinates).T
    transformed_coordinates[:, 0] = transformed_coordinates[:, 0] / transformed_coordinates[:, 2]
    transformed_coordinates[:, 1] = transformed_coordinates[:, 1] / transformed_coordinates[:, 2]
    transformed_coordinates[:, 2] = transformed_coordinates[:, 2] / transformed_coordinates[:, 2]

    transformed_coordinates = transformed_coordinates[:, 0:2].astype(np.int32)
    x_tr, y_tr = transformed_coordinates[:, 0], transformed_coordinates[:, 1]
    x_orig, y_orig = coordinates[:, 0], coordinates[:, 1]
    im[(y_tr, x_tr)] = image[(y_orig, x_orig)]

    return im

# call the warpImage function
logo_warp = warpImage(logo, target_transform, (im.shape[1], im.shape[0]))

'''
Question 3.c: alpha-blend the warped logo and state farm center image

im = logo * alpha_logo + target * (1 - alpha_logo)

Hints:
    - try to avoid for-loop. You could either use numpy's tensor broadcasting or explicitly call np.repeat / np.tile
'''

# --------------------------- Begin your code here ---------------------------------------------

target_alpha = (im * (1 - logo_warp[:, :, 3])[:,:,None])
logo_alpha = (logo_warp * logo_warp[:, :, 3][:,:,None])
im = logo_alpha + target_alpha
#
# # --------------------------- End your code here   ---------------------------------------------
#
plt.clf()
plt.imshow(im)
plt.title("Blended Image")
plt.show()

# dump the results for autograde
outfile = 'solution_homography.npz'

# Downsize file before saving
im = (im * 255).astype(np.uint8)
np.savez(outfile, corners_court, H, target_transform, logo_warp, im)