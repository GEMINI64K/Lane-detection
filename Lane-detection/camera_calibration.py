import cv2
import numpy as np
import glob

# nx = 6
# ny = 4

# img = cv2.imread('checker_board_image.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

# if ret == True:
#     cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
#     #cv2.imshow('camera calibration',img)


objpoints = []
imgpoints = []

images = glob.glob("{}/*".format("camera_cal"))

objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

for image in images:
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(img,(9,6))
    if ret: 
        imgpoints.append(corners)
        objpoints.append(objp)
    shape = (img.shape[1], img.shape[0])
    ret, mtx, dist, _,_ = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)
    
def undistort(distorded_image):
    return cv2.undistort(distorded_image, mtx, dist, None, mtx)

img1 = cv2.imread("camera_cal/calibration18.jpg ")

output = undistort(img1)

cv2.imshow('image',img1)
cv2.imshow('calibrated image',output)

cv2.waitKey(0)
cv2.destroyAllWindows()