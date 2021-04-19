import cv2
import numpy as np
import matplotlib.pyplot as plt
from sonar_display import show_sonar
## use this script to annotate human location from raw image
def readline(line):
    line = line.split()
    line = list(map(int, line))
    angle = line[0]
    data = line[1:]
    return angle, data
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
    return None
def sonar2pool(a, angle, distance):
    if a == 1:
        offset = (0, 5.02)
        rotate = 163
    else:
        offset = (7.2, 0)
        rotate = 95
    angle_pool = (angle - rotate)/200*np.pi
    X = offset[0] + np.cos(angle_pool)*distance
    Y = offset[1] + np.sin(angle_pool)*distance
    return [X,Y]

if __name__=='__main__':
    # files = ["third_dataset/sonar_1/reference.txt", "third_dataset/sonar_2/reference.txt"]
    # for file in files:
    #     print(file)
    #     f = open(file, 'r')
    #     lines = f.readlines()
    #     sonar_img = np.zeros((500, 400))
    #     for i in range(len(lines)):
    #         angle, data = readline(lines[i])
    #         if len(data) == 0:
    #             continue
    #         sonar_img[:, angle] = data
    #     show_sonar(sonar_img, 20)
    #     plt.show()

    cap = cv2.VideoCapture("third_dataset/walk.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    #cameraMatrix = np.array([[3037.3, 0, 2031.5], [0, 3030.1, 1485.3], [0, 0, 1]])
    cameraMatrix = np.eye(3)
    #distCoeffs = np.array([0.1342, -0.3982, 0, 0])
    distCoeffs = np.array([0, 0, 0, 0])
    imageLoc = []
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", frame)
    cv2.waitKey(0)
    imageLoc = np.array(imageLoc, dtype='float32')
    # imageLoc = np.array([[707, 375], [342, 870], [1086, 988], [1420, 263]], dtype='float32')
    worldLoc = np.array([[7, 0, 0], [0, 2.5, 0], [0, 5, 0], [20, 5, 0]], dtype='float32')
    (_, rvec, tvec) = cv2.solvePnP(worldLoc, imageLoc, cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    rotationMatrix, _ = cv2.Rodrigues(rvec)
    rotationMatrix = np.asmatrix(rotationMatrix)
    for i in range(3 * int(cap.get(5)), int(cap.get(7)), 3 * int(cap.get(5))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        imageLoc = []
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
        cv2.imshow("image", frame)
        cv2.waitKey(0)
        print(imageLoc)
        imageOrigin = np.array([[imageLoc[0][0], imageLoc[1][0]],[imageLoc[0][1], imageLoc[0][1]],[1, 1]], dtype='float32')
        print(imageOrigin.shape)
        leftMat = rotationMatrix.I * np.asmatrix(cameraMatrix).I * imageOrigin
        rightMat = rotationMatrix.I * tvec
        s = rightMat[2, 0]/ leftMat[2, 0]
        print(rotationMatrix.I * (s * np.asmatrix(cameraMatrix).I *imageOrigin - tvec ))
    cap.release()
    cv2.destroyAllWindows()

    # for i in range(0, int(cap.get(7)), int(cap.get(5))):
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    #     ret, frame = cap.read()
    #     cv2.imshow('frame', frame)
    #     cv2.waitKey(0)
    #
    # cap.release()
    # cv2.destroyAllWindows()

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

