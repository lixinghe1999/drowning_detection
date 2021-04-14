import cv2
import numpy as np
## use this script to annotate human location from raw image
def transformPoint(current, transformation):
    x = current[:, 0] * transformation[0, 0] + current[:, 1] * transformation[0,1] + transformation[0,2]
    y = current[:, 0] * transformation[1, 0] + current[:, 1] * transformation[1,1] + transformation[1,2]
    z = current[:, 0] * transformation[2,0] + current[:, 1] * transformation[2,1] + transformation[2,2]
    x /= z
    y /= z
    return np.concatenate((x,y))
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        imageLoc.append([x, y])
if __name__=='__main__':
    rawchessboard = cv2.imread("images/IMG_0305.jpg")
    #cameraMatrix = np.array([[3037.3, 0, 2031.5], [0, 3030.1, 1485.3], [0, 0, 1]])
    cameraMatrix = np.eye(3)
    #distCoeffs = np.array([0.1342, -0.3982, 0, 0])
    distCoeffs = np.array([0, 0, 0, 0])
    imageLoc = []
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", rawchessboard)
    cv2.waitKey(0)
    imageOrigin = np.array([[imageLoc[-1][0]], [imageLoc[-1][1]], [1]])
    imageLoc = np.array(imageLoc[:-1], dtype='float32')
    #worldLoc = np.array([[0, 0, 0], [10.5, 0, 0],[21, 0, 0],[21, 7.35,0],[21,14.7,0],[10.5, 14.7, 0],[0, 14.7, 0],[0, 7.35, 0]], dtype='float32')
    #worldLoc = np.array([[0, 0, 0], [21, 0, 0],[21, 14.7,0],[0, 14.7, 0]], dtype='float32')
    (_, rvec, tvec) = cv2.solvePnP(worldLoc, imageLoc, cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    rotationMatrix, _ = cv2.Rodrigues(rvec)
    rotationMatrix = np.asmatrix(rotationMatrix)
    leftMat = rotationMatrix.I * np.asmatrix(cameraMatrix).I * imageOrigin
    rightMat = rotationMatrix.I * tvec
    s = rightMat[2, 0]/ leftMat[2, 0]
    print(rotationMatrix.I * (s * np.asmatrix(cameraMatrix).I *imageOrigin - tvec ))


    # img = cv2.undistort(rawchessboard, cameraMatrix, distCoeffs, None, None)
    # imageLoc = []
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    # cv2.imshow("image", img)
    # cv2.waitKey(0)
    # imageOrigin = np.array([[imageLoc[-1][0], imageLoc[-1][1]]])
    # imageLoc = np.array(imageLoc[:-1], dtype='float32')
    # worldLoc = np.array([[0,0],[21,0],[21,14.7],[0,14.7]], dtype='float32')
    # image2World = cv2.getPerspectiveTransform(imageLoc, worldLoc)
    # world2Image = np.asmatrix(image2World).I
    # print(transformPoint((imageOrigin), image2World))

