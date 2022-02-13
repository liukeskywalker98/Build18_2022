'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction 
    using scatter
'''
'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
from helper import camera2
from submission import (eightpoint, essentialMatrix,
                       triangulate, epipolarCorrespondence)
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

def reconstruct(pts1, pts2, K1, K2, x1=np.array(), y1=np.array()):
    '''
    corres_path = os.path.join("../data", "some_corresp.npz")
    corres = np.load(corres_path)
    pts1 = corres['pts1']
    pts2 = corres['pts2']

    intrinsics_path = os.path.join("../data", "intrinsics.npz")
    intrinsics = np.load(intrinsics_path)
    K1 = intrinsics['K1']
    K2 = intrinsics['K2']

    coords_path = os.path.join("../data", "templeCoords.npz")
    coords = np.load(coords_path)
    x1 = coords['x1']
    y1 = coords['y1']
    '''
    image1_path = os.path.join("../data", "im1.png")
    img1 = Image.open(image1_path)
    img1 = np.array(img1).astype(np.float32)/ 255
    M = np.amax(img1.shape)

    image2_path = os.path.join("../data", "im2.png")
    img2 = Image.open(image2_path)
    img2 = np.array(img2).astype(np.float32)/ 255

    F = eightpoint(pts1, pts2, M)
    E = essentialMatrix(F, K1, K2)
    M2s = camera2(E)
    M1 = np.eye(3, 4)
    C1 = K1 @ M1

    valid_M2 = None
    valid_C2 = None
    valid_count = 0
    # Try out each M2
    for i in range(4):
        M2 = M2s[:, :, i]  
        C2 = K2 @ M2
        P, err = triangulate(C1, pts1, C2, pts2)
        
        # If any points are behind the cameras (negative Z), the view is invalid
        P_homog = np.concatenate((P, np.ones((P.shape[0], 1))), axis = 1)
        invalid1 = (P_homog @ C1.T)[:, 2] < 0
        negative_present1 = np.any(invalid1)
        invalid2 = (P_homog @ C2.T)[:, 2] < 0
        negative_present2 = np.any(invalid2)
        
        if not negative_present1 and not negative_present2:
            print("Valid M2!")
            valid_M2 = M2
            valid_C2 = C2
            valid_count +=1

    if x1.shape[0] > 0:
        # Find the points in the camera 2 frame
        x2 = np.zeros(x1.shape)
        y2 = np.zeros(y1.shape)
        for i in range(len(x1)):
            x = x1[i, 0]
            y = y1[i, 0]
            x2[i], y2[i] = epipolarCorrespondence(img1, img2, F, x, y)
        
        pts1 = np.concatenate((x1, y1), axis = 1)
        pts2 = np.concatenate((x2, y2), axis = 1)

    # Get the real world points
    results, err = triangulate(C1, pts1, valid_C2, pts2)

    # Plot the points
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.scatter(results[:, 0], results[:, 1], results[:, 2],
            marker = '.', c= 'r')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

    result_path = os.path.join(os.pardir, "results", "q4_2.npz")
    #np.savez(result_path, F = F, M1 = M1, M2 = valid_M2, C1 = C1, C2 = valid_C2)
    return results, E
