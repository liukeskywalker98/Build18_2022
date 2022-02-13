"""
Homework4.
Replace 'pass' by your implementation.
"""

import numpy as np
import os
from scipy.signal.windows import gaussian

#from helper import displayEpipolarF
from util import refineF

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    point_count = pts1.shape[0]

    # Scale the points to -1 to 1, keep T.
    ones = np.ones((point_count, 1))
    T = np.array([[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]])
    # Homogenize
    pts1_homo = np.concatenate((pts1, ones), axis = 1)
    pts2_homo = np.concatenate((pts2, ones), axis = 1)
    pts1_homo = pts1_homo @ T.transpose()
    pts2_homo = pts2_homo @ T.transpose()

    # Format the A matrix: potential issue: flipped order of points
    x1 = pts1_homo[:, :1]
    y1 = pts1_homo[:, 1:2]
    x2 = pts2_homo[:, :1]
    y2 = pts2_homo[:, 1:2]
    ones = np.ones((point_count, 1))
    A = np.concatenate((x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, ones),
                       axis = 1)

    # Find the SVD: singular column might be transposed
    u, s, vh = np.linalg.svd(A)
    F_column = vh[-1, :]

    # Build F
    F = np.reshape(F_column, (3, 3))
    u, s, vh = np.linalg.svd(F)
    s[2] = 0 # zero out botom right eigenvalue of sigma matrix
    F = (u * s) @ vh

    # Refine
    F = refineF(F, pts1_homo[:, :2], pts2_homo[:, :2])
    # Enforce linalg error
    assert(np.linalg.matrix_rank(F) < 3)
    #Rescale
    F = T.transpose() @ F @ T

    # Save to file
    result_path = os.path.join("../results", "q2_1.npz")
    np.savez(result_path, F = F, M = M)
    return F

'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    return K2.T @ F @ K1


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    N = pts1.shape[0]
    # Make 3rd order tensors to parallelize the triangulation
    C1_tens = np.repeat(C1.reshape(1, C1.shape[0], C1.shape[1]), N, axis = 0)
    C2_tens = np.repeat(C2.reshape(1, C2.shape[0], C2.shape[1]), N, axis = 0)
    pts1_tens = pts1.reshape((N, 2, 1))
    pts2_tens = pts2.reshape((N, 2, 1))

    # Build A matrix
    z1 = pts1_tens @ C1_tens[:, 2:, :]
    z2 = pts2_tens @ C2_tens[:, 2:, :]  
    non_homogenous1 = C1_tens[:, :2, :] - z1
    non_homogenous2 = C2_tens[:, :2, :] - z2
    A = np.concatenate((non_homogenous1, non_homogenous2), axis = 1)

    # Decompose using SVD and find the right singular vector to get w_i
    
    u, s, vh = np.linalg.svd(A)
    X = vh[:, -1, :]
    X = X.reshape((N, 4))
    P = X[:, :3] / X[:, 3:]

    # Compute reprojection error: simply the square difference across all 
    # elements and dimensions
    P_homog = np.concatenate((P, np.ones((P.shape[0], 1))), axis = 1)
    pts1_hat = P_homog @ C1.T
    pts1_hat_NH = pts1_hat[:, :2] / pts1_hat[:, 2:]
    diff1 = pts1 - pts1_hat_NH

    norm1 = np.square(diff1)
    err1 = np.sum(norm1)

    pts2_hat = P_homog @ C2.T
    pts2_hat_NH = pts2_hat[:, :2] / pts2_hat[:, 2:]
    diff2 = pts2 - pts2_hat_NH
    norm2 = np.square(diff2) 
    err2 = np.sum(norm2)

    err = err1 + err2
    return P, err

'''
Build a linear range with reflection for window selection
    Input:  center, the x or y coordinate where we center the window
            width, the width of the window
            limit, the width/height limit of the image
    Output: window_range, a (2*width+1) vector containing a linear range of 
            coordinates. Overhang values are replaced with reflections.
'''
def get_window_range(center, width, limit):
    left_edge = center - width
    if left_edge < 0:
        left_normal = np.arange(0, center)
        left_reflect = np.arange(-left_edge - 1, -1, -1)
        left_window = np.concatenate((left_reflect, left_normal))
    else:
        left_window = np.arange(left_edge, center)

    right_edge = center + width + 1
    if right_edge > limit:
        overhang = right_edge - limit
        right_normal = np.arange(center, limit)
        right_reflect = np.arange(limit -1, limit - overhang - 1, - 1)
        right_window = np.concatenate((right_normal, right_reflect))
    else:
        right_window = np.arange(center, right_edge)

    window_range = np.concatenate((left_window, right_window))
    return window_range

'''
Builds a window of points from an image
    Input:  x1: the x center of the window
            y1: the y center of the window
            window_size: in pixels
            image: the image matrix to sample from
    Output: window: a (2 * window_size + 1) ^2 matrix of sample pixels
            from image
'''
def build_window(x1, y1, window_size, image):
    rows = get_window_range(y1, window_size, image.shape[0])
    cols = get_window_range(x1, window_size, image.shape[1])
    row, col = np.meshgrid(rows, cols)
    window = image[row, col]
    return window

'''
Given the equation of the epipolar line on image 2, return a vector of points
along which to find matches
    Input:  line_vector: vector defining the epipolar line ax2 + by2 + c = 0
            x: the x coordinate of the closest point from camera 1
            y: the y coordinate of the closest point from camera 1
            search_range: the range over which to find a match
            x_limit: the width of the image
            y_limit: the height of the image
    Output: x_vals: the set of x_points along the epipolar line to search over
            y_vals: the set of y points...
Note: the algorithm clips the search points by the limits of the image.
'''
def raster(line_vector, x, y, search_range, x_limit, y_limit):
    slope = - line_vector[0] / line_vector[1]
    intercept = -line_vector[2] / line_vector[1]
    x_vals = None
    y_vals= None

    if slope < 1:
        # Use x indexing
        x_vals = np.arange(x - search_range, x + search_range + 1)
        y_vals = np.int_(np.around(slope * x_vals + intercept))
    else:
        # Use y indexing
        y_vals = np.arange(y - search_range, y + search_range + 1)
        x_vals = np.int_(np.around((y_vals - intercept) / slope))
    
    # Filter out points that are out of bounds
    valid_x = np.logical_and(x_vals >= 0, x_vals < x_limit)
    valid_y = np.logical_and(y_vals >= 0, y_vals < y_limit)
    valid = np.logical_and(valid_x, valid_y) # both x and y valid

    x_vals = x_vals[valid]
    y_vals = y_vals[valid]
    return x_vals, y_vals

'''
Uses dot product definition to find the closest point on the epipolar line in
image 2 to the original image point in image 1
    Input:  x: the x coordinate of the point from camera 1
            y: the y coordinate of the point from camera 1
            line: the a,b,c vector defining the epipolar line ax2 + by2 + c = 0
    Output: x2: the x coordinate of the closest point on the epipolar line
            y2: the y coordinate .... 
'''
def get_closest_point(x, y, line):
    a = line[0]
    b = line[1]
    c = line[2]
    closest_x = (b ** 2 * x - a * c - a * b * y) / (a ** 2 + b ** 2)
    closest_y = - a / b * closest_x - c / b
    return np.int_(np.around(closest_x)), np.int_(np.around(closest_y))

'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Hyper parameters:
    window_size = 5
    search_range = int(im1.shape[1] / 20)
    
    # Find the im2 epipolar line equation
    p1 = np.array([[x1], [y1], [1]])
    line_coeffs = F @ p1 # this is the [a,b,c] vector, where ax2 + by2 + c = 0
    # We can find points using by2 = -ax2 - c => y_2 = -a/b x2 - c/b
    x_limit = im2.shape[1]
    y_limit = im2.shape[0]

    # Reference window from image 1
    ref_window = build_window(x1, y1, window_size, im1)
    
    # Build guassian filter of radius window_size
    length = window_size * 2 + 1
    gauss_1D = gaussian(length, window_size / 4).reshape((length, 1))
    gauss_window = np.reshape(gauss_1D @ gauss_1D.T, (length, length, 1))

    # Find the closest point to the old point
    x_close, y_close = get_closest_point(x1, y1, line_coeffs)

    min_err = 10000000
    min_x2 = x1
    min_y2 = y1

    # Get coordinate points on the epipolar line in a search range: raster
    x_vals, y_vals = raster(line_coeffs, x_close, y_close, 
                            search_range, x_limit, y_limit)

    # For every point on the search segment:
    for i in range(len(x_vals)):
        # Compute window score
        x2 = x_vals[i]
        y2 = y_vals[i]
        window = build_window(x2, y2, window_size, im2)
        diff = ref_window - window
        square = np.square(diff)
        weight = square * gauss_window
        err = np.sum(weight)
        
        # Store minimum
        if err < min_err:
            min_err = err
            min_x2 = x2
            min_y2 = y2
    
    # Return minimum pts
    return min_x2, min_y2

'''
Q5.1: Extra Credit RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M, nIters=1000, tol=0.42):
    # Replace pass by your implementation
    pass

'''
Q5.2:Extra Credit  Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    pass

'''
Q5.2:Extra Credit  Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    pass

'''
Q5.3: Extra Credit Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and 
            estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    pass

'''
Q5.3 Extra Credit  Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    pass
