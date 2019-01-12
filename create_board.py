import cv2
aruco = cv2.aruco

# Generation of checker board
dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
parameters = aruco.DetectorParameters_create()
board = aruco.CharucoBoard_create(4, 4, 0.045, 0.03, dictionary) # squaresX, squaresY, squareLength, markerLength, dictionary
pixel_per_onebox = 256
boardImage = 0
boardImage = board.draw((pixel_per_onebox*5,pixel_per_onebox*8), boardImage, 0, 1)
cv2.imwrite("charuco_board.bmp", boardImage)
cv2.imshow("charuco", boardImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
