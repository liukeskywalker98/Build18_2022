import cv2
import numpy as np

from visualize import reconstruct
'''
The goal of camera.py is to test concepts of monocular camera-based mapping.

For a monocular camera system, doing 3D reconstruction is tricky. Since there is
only one perspective, the robot must rely on egomotion to get 3D data.

How it works:
camera.py records and stores images as the "robot" moves. Each time a frame is 
captured, camera.py computes a 3D reconstruction of the scene using previously
seen images. camera.py also attempts to localize the robot by returning
information about its movement (epipolar motion) between frames. The basics of
epipolar geometry are a solved problem. The challenges are in implementing an 
efficient implementation.

Big questions to answer:
Which images are used for 3D reconstruction? Is reconstruction done every frame?
This seems to be a bad solution due to accumulating errors. It's a good idea to
refer to images "long in the past" and try to compute the geometry. However, at
this extreme is also a bad solution. We could compute a 3D reconstruction from
the current to all previous frames. This would give us maximum information about
the environment. It would also crash within seconds due to the time and memory
complexity of storing, then doing online epipolar math on an exponentially 
growing set of image pairs.

A solution is somewhere in between: the algorithm could choose to compute only 
after a set "period" of frames. This has a few advantages. It means that the 
robot could accrue significant motion before it tried to do 3D reconstruction.
Larger optical flow provides a larger virtual baseline, which provides better 
information about range. This also gives time for the computer to compute its
epipolar math, which can take a lot of time. The downsides are that the period
becomes a tunable hyperparameter that creates problems. How far should the robot
go before computing? What if the robot makes sudden turns which put the previous
completely out of frame? This means no correspondence can be computed, and
therefore the robot breaks its localization. This means to maintain high
accuracy state estimates, the robot must constrain its movements or set periods
really low. Even with this solution, it is also drifty in that errors can 
accumulate.

Another possible solution is to pick which images to use for 3D reconstruction.
The previous image might not make sense as the image to compare to. Instead, the
robot might store some "key" images and then pick which one is most relevant. 
This solution can help correct for drift in the estimates. If the robot looks at
features found in the original image, for example, it might be able to
completely eliminate all drift (within image processing error).
'''

imageCache = np.array((0,1,1))

def pickImage():
    return imageCache[:,:,0]

def reconstruct():
    pass

def retrievePose(E):
    # Retrieve translation, up to sign
    u,s,v = np.linalg.svd(E)
    t = u[-1,:] # could be positive or negative of real translation

    # Retrieve 2 possible rotations
    possibleR = np.array(2,3,3)
    rotationPos90 = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    rotationNeg90 = rotationPos90.T
    validIndex = 0
    for i in range(4):
        if i // 2 == 1:
            rotation = rotationPos90
        else:
            rotation = rotationNeg90
        candidate = (-1 ** i) * u @ rotation @ v.T
        
        if np.linalg.det(candidate) > 0:
            possibleR[i,:,:] = candidate
            validIndex += 1
    return t, possibleR

# Get a video feed from the laptop's camera
vc = cv2.VideoCapture(0) #0 -> index of camera
if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

cv2.namedWindow("Kevin's face")

# Load intrinsic matrix of the camera
# K = loadIntrinsic()

while rval:
    # TODO: Pick the image to be used for comparison
    #img1 = pickImage()

    # TODO: Find correspondences
    #(pts1, pts2) = findCorrespondences(img1, frame)

    # Do 3D Reconstruction: get a set of 3D points
    points, E = reconstruct(pts1, pts2, K, K)
    
    # TODO: Cluster to find objects: want number of objects and distance to each
    # objects = cluster(points)

    # Get the pose change from the 3D reconstruction
    Rs, ts = retrievePose(E)
    
    # TODO: Use odometry or some other metric to filter out one of the rotations
    # R, t = filterRT(Rs, ts, stateEstimate)

    # TODO: Update the state estimate of the robot
    # stateEstimate = fuse(stateEstimate, R, t)

    # TODO: Update the map
    # updateMap(map, objects) 

    # Get a new frame, optionally break
    rval, frame = vc.read()
    cv2.imshow("Kevin's face", frame)
    key = cv2.waitKey(20)
    if key == 28: # if escape was pressed
        cv2.destroyWindow("Kevin's face")
        break

