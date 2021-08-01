import cv2
import numpy as np
import argparse
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import time
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
    if a == 0:
        offset = (0, 5.02)
        rotate = 163
    else:
        offset = (7.2, 0)
        rotate = 95
    angle_pool = (angle - rotate)/200*np.pi
    X = offset[0] + np.cos(angle_pool)*distance
    Y = offset[1] + np.sin(angle_pool)*distance
    return [X,Y]
def txt2loc(f, w, h):
    f_ob = open(f)
    lines = f_ob.readlines()
    result = [[], [], []]
    for line in lines:
        line = line.split()
        if int(line[0]) != 0 or (float(line[1]) + float(line[2])) < 0.5:
            continue
        else:
            result[0].append(float(line[1]) * w)
            result[1].append(float(line[2]) * h)
            result[2].append(1)
    return result
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Control the sonar')
    parser.add_argument('--mode', action="store", required=False, type=int, default=1,
                        help="0-yolo detection prepare, 1-localization")
    args = parser.parse_args()
    if args.mode == 0:
        for f in ['2021-04-15-11-18-08', '2021-04-15-11-30-31']:
            end_time = datetime.strptime(f, "%Y-%m-%d-%H-%M-%S")
            cap = cv2.VideoCapture("third_dataset/" + f + '.mp4')
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # save as jpg
            for i in range(0, int(cap.get(7)), 3 * int(cap.get(5))):
                now_time = end_time - timedelta(seconds=int(i/int(cap.get(5))))
                now_time_str = now_time.strftime("%Y-%m-%d-%H-%M-%S")
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                cv2.imwrite('third_dataset/images/'+now_time_str + '.jpg' , frame)
    #python detect.py --source ../third_dataset/images --weights yolov5m.pt --conf 0.25 --save-txt
    else:
        cameraMatrix = np.array([[2803.5, 0, 1837.4], [0, 2807.8, 1394.1], [0, 0, 1]])
        distCoeffs = np.array([-0.0219, 0.0142, 0, 0])
        #cameraMatrix = np.eye(3)
        #distCoeffs = np.array([0, 0, 0, 0])
        path = "third_dataset/labels/"
        labels = os.listdir(path)
        frame = cv2.imread("third_dataset/images/2021-04-15-11-04-50.jpg")
        h, w, c = np.shape(frame)
        # imageLoc = []
        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
        # cv2.imshow("image", frame)
        # cv2.waitKey(0)
        # imageLoc = np.array(imageLoc, dtype='float32')
        # print(imageLoc)
        imageLoc = np.array([[ 973. , 296.],[1141. , 254.],[1371.,  262.],[ 988.,  980.],[ 556.,  920.],[ 146.,  866.]], dtype='float32')
        worldLoc = np.array([[13.13, 0, 0], [24.13, 0, 0], [24.13, 3.53, 0], [0, 3.53, 0], [0, 2.78, 0], [0, 2.03, 0]], dtype='float32')
        (_, rvec, tvec) = cv2.solvePnP(worldLoc, imageLoc, cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        rotationMatrix, _ = cv2.Rodrigues(rvec)
        rotationMatrix = np.asmatrix(rotationMatrix)

        for f in labels:
            print(f)
            imageOrigin = np.array(txt2loc(path + f, w, h), dtype='float32')
            leftMat = rotationMatrix.I * np.asmatrix(cameraMatrix).I * imageOrigin
            rightMat = rotationMatrix.I * tvec
            s = rightMat[2, 0]/ leftMat[2, 0]
            result = rotationMatrix.I * (s * np.asmatrix(cameraMatrix).I *imageOrigin - tvec )
            #result = imageOrigin
            for i in range(np.shape(result)[1]):
                #if result[0, i] < 10 and result[0, i] > 0 and result[1, i] < 6 and result[1, i] > 0:
                plt.scatter(result[0, i], result[1, i])
            # plt.scatter(list(result[0, :]), list(result[1, :]))
            # print(list(result[0, :]), list(result[1, :]))
            # plt.xlim([0, 10])
            # plt.ylim([0, 6])
        plt.xlim([0, 20])
        plt.ylim([0, 6])
        plt.show()
        cv2.destroyAllWindows()

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

