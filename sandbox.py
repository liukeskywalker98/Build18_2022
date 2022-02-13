from helper import displayEpipolarF, epipolarMatchGUI
from submission import eightpoint
import os
from PIL import Image
import numpy as np

if __name__ == '__main__':
    corres_dir = os.path.join("../data", "some_corresp.npz")
    corres = np.load(corres_dir)
    pts1 = corres['pts1']
    pts2 = corres['pts2']
    
    image1_path = os.path.join("../data", "im1.png")
    img1 = Image.open(image1_path)
    img1 = np.array(img1).astype(np.float32)/ 255
    
    image2_path = os.path.join("../data", "im2.png")
    img2 = Image.open(image2_path)
    img2 = np.array(img2).astype(np.float32) / 255
    
    # Question 2.1
    M = np.amax(img1.shape)
    F = eightpoint(pts1, pts2, M)
    #displayEpipolarF(img1, img2, F)

    # Question 4.1
    epipolarMatchGUI(img1, img2, F)
    pts1 = np.array([[120.61827956989245, 211.49139784946237], 
           [64.6010752688172, 132.76451612903236],
           [82.76881720430103, 279.6204301075269],
           [123.6462365591398, 323.525806451613],
           [205.4010752688172, 209.97741935483873],
           [237.194623655914, 156.98817204301076],
           [314.4075268817205, 222.089247311828],
           [371.9387096774194, 278.10645161290324],
           [325.005376344086, 341.6935483870968],
           [441.58172043010757, 225.11720430107528],
           [479.431182795699, 100.97096774193551],
           [514.2526881720431, 226.63118279569892],
           [462.7774193548388, 385.5989247311828],
           [223.5688172043011, 371.9731182795699]])
    pts2 = np.array([[119, 179], [65, 119], [82, 282], [123, 313], [204, 186],
                     [235, 152], [313, 201], [373, 272], [330, 350], [440, 195],
                     [471, 100], [513, 202], [471, 353], [226, 384]])    
    result_path = os.path.join("../results", "q4_1.npz")
    np.savez(result_path, F = F, pts1 = pts1[:, :], pts2 = pts2[:, :])
    