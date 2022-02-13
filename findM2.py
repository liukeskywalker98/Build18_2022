'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
from helper import camera2
from submission import eightpoint, essentialMatrix, triangulate
from PIL import Image
import os
import numpy as np

if __name__ == '__main__':
    corres_path = os.path.join("../data", "some_corresp.npz")
    corres = np.load(corres_path)
    pts1 = corres['pts1']
    pts2 = corres['pts2']

    intrinsics_path = os.path.join("../data", "intrinsics.npz")
    intrinsics = np.load(intrinsics_path)
    K1 = intrinsics['K1']
    K2 = intrinsics['K2']

    image1_path = os.path.join("../data", "im1.png")
    img1 = Image.open(image1_path)
    img1 = np.array(img1).astype(np.float32)/ 255
    M = np.amax(img1.shape)

    F = eightpoint(pts1, pts2, M)
    E = essentialMatrix(F, K1, K2)
    M2s = camera2(E)
    M1 = np.eye(3, 4)

    valid_M2 = None
    valid_C2 = None
    valid_P = None

    valid_err = 0
    valid_count = 0
    # Try out each M2
    for i in range(4):
        M2 = M2s[:, :, i]
        C1 = K1 @ M1
        C2 = K2 @ M2
        P, err = triangulate(C1, pts1, C2, pts2)
        
        # If any points are behind the cameras (negative Z), the view is invalid
        P_homog = np.concatenate((P, np.ones((P.shape[0], 1))), axis = 1)
        invalid1 = (P_homog @ C1.T)[:, 2] < 0
        negative_present1 = np.any(invalid1)
        invalid2 = (P_homog @ C2.T)[:, 2] < 0
        negative_present2 = np.any(invalid2)
        
        if not negative_present1 and not negative_present2:
            valid_M2 = M2
            valid_C2 = C2
            valid_P = P
            valid_count +=1
            valid_err = err

    result_path = os.path.join(os.pardir, "results", "q3_3.npz")
    np.savez(result_path, M2 = valid_M2, C2 = valid_C2, P = valid_P)
