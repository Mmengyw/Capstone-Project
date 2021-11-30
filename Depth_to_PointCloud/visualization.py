import cv2
import numpy as np
import matplotlib.pyplot as plt
import CoordFrame
import vpython as vp
from hw5 import FilterByEpipolarConstraint, least_squares_triangulate

# Pose of drone body relative to reference frame
# pos (x, y, z), quat (w, x, y, z)
g_b_t = [
    [4.688319, -1.786938, 0.783338, 0.534108, -0.153029, -0.827383, -0.082152],
    [3.860339, 0.311198, 0.9168, 0.537515, -0.255038, -0.789015, -0.153293],
    [-0.905146, 4.210864, 0.937239, 0.545842, -0.138205, -0.822061, -0.084689],
    [-2.625492, 7.287188, 0.783302, 0.497072, -0.425954, -0.695357, -0.296587],
    [0.777645, 5.20188, 0.786036, 0.207405, -0.761252, -0.323366, -0.522411]
]

# Transform from body to camera frames
g_b1 = np.array([
    [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
    [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
    [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
    [0.0, 0.0, 0.0, 1.0]
])
g_b2 = np.array([
    [0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556],
    [0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024],
    [-0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038],
    [0.0, 0.0, 0.0, 1.0]
])

# We only need to calculate these once
g_21 = np.matmul(np.linalg.inv(g_b2), g_b1)
R = g_21[:3, :3]
T = g_21[:3, 3].reshape(3, 1)

# Intrinsic camera matrices
K_1 = np.array([
    [458.654, 0, 367.215],
    [0, 457.296, 248.375],
    [0, 0, 1]
])
K_2 = np.array([
    [457.587, 0, 379.999],
    [0, 456.134, 255.238],
    [0, 0, 1]
])

# Distortion coefficients
D_1 = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05])
D_2 = np.array([-0.28368365,  0.07451284, -0.00010473, -3.55590700e-05])

# Create a blank canvas for vpython stuff
scene = vp.canvas()

def get_distored_images(i, vis=False):
    """
    Read in distorted images from memory
    """
    left_image = cv2.imread(f"img/cam1_{i + 1}.png")
    right_image = cv2.imread(f"img/cam2_{i + 1}.png")
    if vis:
        distored_stacked = np.hstack((left_image, right_image))
        plt.imshow(distored_stacked)
        plt.title(f"Distorted Stereo Image {i + 1}")
        plt.axis("off")
        plt.show()
    return left_image, right_image

def get_undistorted_images(i, left_image, right_image, vis=False):
    """
    Take distorted images and apply distortion model to undistort image
    """
    left_undistorted = cv2.undistort(left_image, K_1, D_1)
    right_undistorted = cv2.undistort(right_image, K_2, D_2)
    if vis:
        undistored_stacked = np.hstack((left_undistorted, right_undistorted))
        plt.imshow(undistored_stacked)
        plt.title(f"Undistorted Stereo Image {i + 1}")
        plt.axis("off")
        plt.show()
    return left_undistorted, right_undistorted

def get_feature_matches(i, threshold, octaves, vis=False):
    """
    Compute matching point features accross both images
    """
    feature_detector = cv2.BRISK_create(threshold, octaves=octaves)
    # find the keypoints and descriptors with BRISK
    kp1, des1 = feature_detector.detectAndCompute(left_undistorted, None)
    kp2, des2 = feature_detector.detectAndCompute(right_undistorted, None)
    #Does feature matching
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1,des2)
    left_image_matches = np.array([kp1[i.queryIdx].pt for i in matches])
    right_image_matches = np.array([kp2[i.trainIdx].pt for i in matches])
    # Visualize matches
    keypt_left = left_undistorted.copy()
    keypt_right = right_undistorted.copy()
    cv2.drawKeypoints(left_undistorted, kp1, keypt_left)
    cv2.drawKeypoints(left_undistorted, kp1, keypt_right)
    stacked_kpt = np.hstack((keypt_left, keypt_right))
    match_vis1 = cv2.drawMatches(
	        left_undistorted, kp1,
	        right_undistorted, kp2, matches, 0,
	        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    if vis:
        plt.imshow(stacked_kpt)
        plt.title(f"Keypoints in Stereo Image {i + 1}")
        plt.axis("off")
        plt.show()

        plt.imshow(match_vis1)
        plt.title(f"Matches in Stereo Image {i + 1}")
        plt.axis("off")
        plt.show()
    return kp1, kp2, matches, left_image_matches, right_image_matches, match_vis1

def filter_matches(i, threshold, kp1, kp2, matches, vis=False):
    """
    Filter maching point features through geometric constraints
    """
    inlier_mask = np.array(FilterByEpipolarConstraint(K_1, K_2, kp1, kp2, R, T, threshold, matches)) == 1
    left_image_masked = np.pad(left_image_matches[inlier_mask], [(0, 0), (0, 1)], mode='constant', constant_values=1)
    right_image_masked = np.pad(right_image_matches[inlier_mask], [(0, 0), (0, 1)], mode='constant', constant_values=1)

    filtered_matches = [m for m, b in zip(matches, inlier_mask) if b == 1]
    match_vis2 = cv2.drawMatches(
        left_undistorted, kp1,
        right_undistorted, kp2, filtered_matches, 0,
        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    if vis:
        plt.imshow(match_vis2)
        plt.title(f"Filtered Matches in Stereo Image {i + 1}")
        plt.axis("off")
        plt.show()

    return left_image_masked, right_image_masked, filtered_matches, match_vis2

def triangulate_matches(left_image_masked, right_image_masked,vis=False):
    """
    Resolve depth of points with known correspondance
    """
    global scene
    n = left_image_masked.shape[0]
    points_list = []
    for i in range(n):
        x_1 = left_image_masked[i, :].reshape((-1, 1))
        x_2 = right_image_masked[i, :].reshape((-1, 1))
        x_w = least_squares_triangulate(x_1, x_2, R, T, K_1, K_2)
        if x_w is not None:
            x_1_i = x_1[:2].astype('uint32').reshape((1, 2))[:, ::-1]
            x_2_i = x_2[:2].astype('uint32').reshape((1, 2))[:, ::-1]
            point = x_w.flatten().tolist()
            points_list.append(point)

    # Things get way too far away so we are only considering orientation
    g_b_r = g_b
    g_b_r[:3, 3] = np.zeros(3)
    points_list_rot = [vp.vector(*np.matmul(g_b_r, np.array(point + [1]))[:3]) for point in points_list]

    if vis:
        scene.delete()
        scene = vp.canvas()
        # Draw coordinate frames and pointcloud
        scene.forward = vp.vector(-1, -1, -.5)
        scene.up = vp.vector(0, 0, 1)
        scene.range = .2
        CoordFrame.draw(g_b_r, scale=.07, name="B")
        CoordFrame.draw(np.matmul(g_b_r, g_b1), scale=.07, name="1")
        CoordFrame.draw(np.matmul(g_b_r, g_b2), scale=.07, name="2")
        if points_list_rot:
            vp.points(pos=points_list_rot)
        scene.waitfor("redraw")

    return points_list

for i in range(5):
    """
    Loop through images and do fun CV things
    """
    g_b_cur = g_b_t[i]
    g_b = CoordFrame.g_from_vec_quat(g_b_cur[:3], g_b_cur[3:])

    # Load distorted stereo image pair from memory
    left_image, right_image = get_distored_images(i, vis=True)

    # Undistort image using known distortion model
    left_undistorted, right_undistorted = get_undistorted_images(i, left_image, right_image, vis=True)

    # Get correspondances between camera images
    kp1, kp2, matches, left_image_matches, right_image_matches, match_vis1 = get_feature_matches(i, 30, 5, vis=True)

    # Filter correspondances through two-view geometry
    # Will not work until part (c) is completed
    left_image_masked, right_image_masked, filtered_matches, match_vis2 = filter_matches(i, 0.07, kp1, kp2, matches, vis=False)

    # Visualize both original correspondances and filtered correspondances
    #full_matches_vis = True
    #if (full_matches_vis):
        #match_vis = cv2.vconcat([match_vis1, match_vis2])
        #plt.imshow(match_vis)
        #plt.title(f"Original vs Filtered Matches in Stereo Image {i + 1}")
        #plt.axis("off")
        #plt.show()

    # Determine world coordinates of corresponding points.
    # Will not work until part (d) is completed
    triangulate_matches(left_image_masked, right_image_masked, vis=True)
