import numpy as np
np.set_printoptions(precision=3)
import cv2
aruco = cv2.aruco
import pickle

# Generation of checker board
dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
parameters = aruco.DetectorParameters_create()
board = aruco.CharucoBoard_create(4, 4, 0.045, 0.03, dictionary) # squaresX, squaresY, squareLength, markerLength, dictionary
img = board.draw((256*5, 256*8))
calibImages = []
count = 0
while count < 6: # 6 - max count of board shots
    calibImage = cv2.imread('res'+str(count)+'.png')
    count += 1
    if calibImage is None:
        break
    calibImages.append(calibImage)

imgSize = calibImages[0].shape[:2]
allCharucoCorners = []
allCharucoIds = []
charucoCorners, charucoIds = [0,0]

# Detect intersection of the checker board
for calImg in calibImages:
    # Find Aruco markers
    res = aruco.detectMarkers(calImg, dictionary)
    # Find Charuco corners
    if len(res[0])>0:
        res2 = aruco.interpolateCornersCharuco(res[0], res[1], calImg, board)
        if res2[1] is not None and res2[2] is not None and len(res2[1])>3:
            allCharucoCorners.append(res2[1])
            allCharucoIds.append(res2[2])
        cv2.aruco.drawDetectedMarkers(calImg,res[0],res[1])
    img = cv2.resize(calImg, None, fx=0.5, fy=0.5)
    cv2.imshow('calibration image',img)
    cv2.waitKey(0)
cv2.destroyAllWindows()

# Calibration and output the errors
try:
    cal = cv2.aruco.calibrateCameraCharucoExtended(allCharucoCorners,allCharucoIds,board,imgSize,None,None)
except:
    print("can not calibrate ...")

retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsInstrinsics, stdDeviationsExtrinsics, perViewErrors = cal

tmp = [cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsInstrinsics, stdDeviationsExtrinsics]

# Save the parameters
with open('cam_param.pkl', mode='wb') as f:
    pickle.dump(tmp, f, protocol=2)
