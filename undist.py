import numpy as np
import cv2
aruco = cv2.aruco
import pickle

# Generation of checker board
dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
parameters = aruco.DetectorParameters_create()
board = aruco.CharucoBoard_create(4, 4, 0.045, 0.03, dictionary) # squaresX, squaresY, squareLength, markerLength, dictionary

# load coeffs
with open('cam_param.pkl', 'rb') as f:
    camera_param = pickle.load(f)
cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsInstrinsics, stdDeviationsExtrinsics = camera_param

calibImages = []
count = 0
while count < 6:
    calibImage = cv2.imread('res'+str(count)+'.png')
    count += 1
    if calibImage is None:
        break
    calibImages.append(calibImage) # save the loaded image
    
# size of img
h,  w = calibImages[0].shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix,distCoeffs,(w,h),1,(w,h))
for i, before_undistortImg in enumerate(calibImages):
    # undistort
    dst = cv2.undistort(before_undistortImg, cameraMatrix, distCoeffs, None, newcameramtx)
    cv2.imshow('undistort', dst)
    cv2.waitKey(0)
    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    # write the new image
    cv2.imwrite('calibresult'+ str(i) +'.png',dst)
