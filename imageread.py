import cv2
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from sonar_display import show_sonar
import torch

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
        if int(line[0]) != 0 or (float(line[1]) + float(line[2]))<0.7:
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
        for f in ['swim', 'walk']:
            cap = cv2.VideoCapture("third_dataset/" + f + '.mp4')
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # save as jpg
            for i in range(0, int(cap.get(7)), 3 * int(cap.get(5))):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                cv2.imwrite('third_dataset/images/'+str(i)+ '-' + f + '.jpg' , frame)
    # python code python detect.py --source ../third_dataset/images --weights yolov5m.pt --conf 0.25 --save-txt
    else:
        cameraMatrix = np.array([[2803.5, 0, 1837.4], [0, 2807.8, 1394.1], [0, 0, 1]])
        distCoeffs = np.array([-0.0219, 0.0142, 0, 0])
        #cameraMatrix = np.eye(3)
        #distCoeffs = np.array([0, 0, 0, 0])
        for f in ['swim', 'walk']:
            path = "yolov5/runs/detect/exp4/labels/"
            labels = os.listdir(path)
            cap = cv2.VideoCapture("third_dataset/" + f + '.mp4')
            ret, frame = cap.read()
            h, w, c = np.shape(frame)
            imageLoc = []
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
            cv2.imshow("image", frame)
            cv2.waitKey(0)
            #print(imageLoc)
            imageLoc = np.array(imageLoc, dtype='float32')
            # imageLoc = np.array([[707, 375], [342, 870], [1086, 988], [1420, 263]], dtype='float32')
            worldLoc = np.array([[10, 0, 0], [20, 0, 0], [20, 5, 0], [10, 5, 0], [0, 5, 0], [0, 2.5, 0]], dtype='float32')
            (_, rvec, tvec) = cv2.solvePnP(worldLoc, imageLoc, cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            rotationMatrix, _ = cv2.Rodrigues(rvec)
            rotationMatrix = np.asmatrix(rotationMatrix)
            for f in labels:
                imageOrigin = np.array(txt2loc(path + f, w, h), dtype='float32')
                leftMat = rotationMatrix.I * np.asmatrix(cameraMatrix).I * imageOrigin
                rightMat = rotationMatrix.I * tvec
                s = rightMat[2, 0]/ leftMat[2, 0]
                print(rotationMatrix.I * (s * np.asmatrix(cameraMatrix).I *imageOrigin - tvec ))
                break
            cap.release()
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

