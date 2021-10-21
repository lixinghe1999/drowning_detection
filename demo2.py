# one sonar, complementary to demo1
from datetime import datetime, timedelta
import torchvision.transforms as transforms
from models import *
from dataset import SonarDataset
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import gridspec
from math import dist
import itertools
from torch.utils.data import ConcatDataset

def sonar2pool(angle, distance):
    offset = (8, 0)
    rotate = 133
    angle_pool = (angle - rotate)/200*np.pi
    X = offset[0] - np.cos(angle_pool) * distance
    Y = offset[1] + np.sin(angle_pool)*distance
    return [X,Y]
def points_merge(sonar_merge):
    combinations = list(itertools.combinations(range(num_sonar), 2))
    for comb1, comb2 in combinations:
        m, n = len(sonar_merge[comb1][0]), len(sonar_merge[comb2][0])
        if m > 0 and n > 0:
            for i in range(m):
                for j in range(n):
                    if sonar_merge[comb1][2][i] and sonar_merge[comb2][2][j]:
                        if dist(sonar_merge[comb1][2][i][:2], sonar_merge[comb2][2][j][:2]) < 1 and sonar_merge[comb1][2][i][2] == sonar_merge[comb2][2][j][2]:
                            loc1, loc2 = sonar_merge[comb1][2][i], sonar_merge[comb2][2][j]
                            sonar_merge[comb1][0][i].remove()
                            sonar_merge[comb2][0][j].remove()
                            sonar_merge[comb1][1][i].remove()
                            sonar_merge[comb2][1][j].remove()
                            sonar_merge[comb1][0].append(ax1.scatter((loc1[0] + loc2[0])/2, (loc1[1] + loc2[1])/2, s = 40, c ='blue'))
                            sonar_merge[comb1][1].append(ax1.text((loc1[0] + loc2[0])/2, (loc1[1] + loc2[1])/2
                                , sonar_merge[comb1][2][i][2] + '\n' + '(' + str((loc1[0] + loc2[0])/2)[:3] + ',' + str((loc1[1] + loc2[1])/2)[:3] + ')', fontsize=15))
                            sonar_merge[comb1][2].append([(loc1[0] + loc2[0])/2, (loc1[1] + loc2[1])/2, sonar_merge[comb1][2][i][2]])
                            sonar_merge[comb1][2][i], sonar_merge[comb2][2][j] = None, None
    return sonar_merge

if __name__ == '__main__':
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    demo_Norm = transforms.Normalize((47.76188580038861, 47.03154234737973, 46.73351142119235), (23.2561984533074, 22.776042059810912, 22.849482616025732))
    demo_add_Norm = transforms.Normalize((40.42246475383515, 37.80934053193982, 36.68358740495986),
                                         (19.3059531023137, 18.936400210616764, 19.24052921650599))
    transform_demo = transforms.Compose([transforms.ToTensor(), demo_Norm])
    transform_demo_add = transforms.Compose([transforms.ToTensor(), demo_add_Norm])
    dataset = SonarDataset(filename='demo_add.txt', transform=transform_demo_add, inferwname=True)

    demoloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    net = LeNet((36, 24), 3).to(device)
    net.load_state_dict(torch.load("checkpoint/92.59-1.pth"))
    result_map = {0: 'Normal', 1: 'Drowning', 2: 'Drowning', 3: 'Normal'}
    time_slot = []
    prediction = []
    i = 0
    f = open('7/update_schedule.txt', 'r')
    for data in demoloader:
        images, labels, name = data
        images, labels = images.to(device, dtype=torch.float), labels.to(device, dtype=torch.int64)
        #outputs = net(images)
        #_, predicted = torch.max(outputs.data, 1)
        loc = name[0].split('/')[-1].split('_')[2].split('-')
        label = int(name[0].split('/')[-1].split('_')[1])
        angle = (int(loc[0]) + int(loc[1]))/2
        distance = (int(loc[2]) + int(loc[3]))/ 2 / 25
        loc = sonar2pool(angle, distance)
        time_slot.append(name[0].split('/')[-1].split('_')[3].split('.')[0])
        #prediction.append([result_map[predicted.item()], loc])
        prediction.append([result_map[label], loc])
    sort_id = sorted(range(len(time_slot)), key=lambda k: time_slot[k])

    fig = plt.figure()
    fig.tight_layout()
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0.1, wspace=0.1)
    cap1 = cv2.VideoCapture("7/102253.mp4")
    cap2 = cv2.VideoCapture("7/113434.mp4")
    A = [[] for i in range(4)]
    # each sensor will have (0) update schedule (1) point object (2) text object (3) location
    f = open("7/update_schedule.txt").readlines()
    A[0] = [x.strip() for x in f]

    # p1, p2, t1, t2, r1, r2 = [], [], [], [], [], []
    count = 0
    ax1.set_xlim([0, 10])
    ax1.set_xlabel('location X axis/m', fontsize=20)
    ax1.set_ylim([0, 15])
    ax1.set_ylabel('location Y axis/m', fontsize=20)
    ax1.scatter(8, 0, s=100, marker='s',  c='red')
    ax1.text(8, 0.5, 'sonar', fontsize=20)
    ax1.set_aspect(1.25)
    sort_id = sort_id[::2]
    for id in sort_id:
        t = time_slot[id]
        r = prediction[id]
        if t > "2021-09-16-11-34-34":
            t_start = datetime.strptime('2021-09-16-11-34-34', "%Y-%m-%d-%H-%M-%S")
            cap = cap2
        else:
            t_start = datetime.strptime('2021-09-16-10-22-53', "%Y-%m-%d-%H-%M-%S")
            cap = cap1
        if t >= A[0][count] and len(A[1]) > 0:
            ax1.set_title('Visualization of the sonar result\n' + A[0][count])
            seconds = (datetime.strptime(A[0][count], "%Y-%m-%d-%H-%M-%S") - t_start).total_seconds()
            i = round(cap.get(5)) * seconds
            cap.set(cv2.CAP_PROP_POS_FRAMES, i+30)
            for j in range(60):
                ax2.cla()
                ax2.set_xticks([])
                ax2.set_yticks([])
                ret, frame = cap.read()
                if ret and j % 2 == 0:
                    ax2.set_title('RGB video\n' + A[0][count])
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    ax2.imshow(frame)
                    plt.pause(0.001)
            for j in range(len(A[1])):
                if A[3][j]:
                    A[1][j].remove()
                    A[2][j].remove()
            A[1:] = [], [], []
        while (count < (len(A[0])-1)):
            if A[0][count] <= t:
                count = count + 1
            else:
                break
        A[1].append(ax1.scatter(r[1][0], r[1][1], s=40, c='blue'))
        A[2].append(ax1.text(r[1][0], r[1][1], r[0] + '\n' + '(' + str(r[1][0])[:3] + ',' + str(r[1][1])[:3] + ')', fontsize=15))
        A[3].append([r[1][0], r[1][1], r[0]])

