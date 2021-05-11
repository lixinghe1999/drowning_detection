from dataset import  SonarDataset
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
import cv2
def extract_features(img):
    F = []
    threshold = 0.2
    f1 = abs(img[:, :, 1] - img[:, :, 0])
    f_thershold = f1 > threshold
    f1[f_thershold] = 0
    F.append(np.mean(f1, axis=(0, 1)))
    F.append(np.sum(f_thershold!=0, axis=(0,1)))

    f2 = abs(img[:, :, 2] - img[:, :, 1])
    f_thershold = f2 > threshold
    f2[f_thershold] = 0
    F.append(np.mean(f2, axis=(0, 1)))
    F.append(np.sum(f_thershold != 0, axis=(0, 1)))

    # f3 = abs(img[:, :, 2] - img[:, :, 0])
    # f_thershold = f3 > threshold
    # f3[f_thershold] = 0
    # F.append(np.mean(f3, axis=(0, 1)))
    # F.append(np.sum(f_thershold != 0, axis=(0, 1)))
    for j in range(3):
        F.append(np.max(img[:, :, j], axis=(0, 1)))
        F.append(np.mean(img[:, :, j], axis=(0, 1)))

    return F

if __name__ == '__main__':
    Norm = ((52.599049512970446, 52.580069378570286, 52.56049022923118), (15.84267112429285, 15.855886602198726, 15.866821187867181))
    train_dataset = SonarDataset(filename='train.txt')
    valid_dataset = SonarDataset(filename='validate.txt')
    X = []
    y = []
    a = 0
    for i in range(len(train_dataset)):
        img, label = train_dataset[i]
        img = (img-np.array(Norm[0]))/np.array(Norm[1])
        F = extract_features(img)
        X.append(F)
        y.append(label.item())
    clf = svm.SVC()
    clf.fit(X, y)
    correct = 0
    for i in range(len(valid_dataset)):
        img, label = valid_dataset[i]
        img = (img-np.array(Norm[0]))/np.array(Norm[1])
        F = extract_features(img)
        if clf.predict([F]) == label.item():
            correct = correct + 1
    print(correct / len(valid_dataset))

