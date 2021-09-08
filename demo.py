# simulateneously show (1) detection result with classification result (2) image at the same time
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

def sonar2pool(sonar, angle, distance):
    if sonar == 'sonar_1':
        offset = (4.8, 0)
        rotate = 118.5
    else:
        offset = (8, 0)
        rotate = 105
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
    Norm = transforms.Normalize((47.76188580038861, 47.03154234737973, 46.73351142119235), (23.2561984533074, 22.776042059810912, 22.849482616025732))
    transform = transforms.Compose([transforms.ToTensor(), Norm])
    demo_transform = transform
    demo_dataset = SonarDataset(filename='demo.txt', transform=demo_transform, inferwname=True)
    demoloader = torch.utils.data.DataLoader(demo_dataset, batch_size=1, shuffle=False)
    net = LeNet((36, 24), 3).to(device)
    net.load_state_dict(torch.load("checkpoint/95.37-6.pth"))
    result_map = {0: 'Normal', 1: 'Drowning', 2: 'Drowning'}
    time_slot = []
    prediction = []
    i = 0
    f1 = open('6/sonar_1/update_schedule.txt', 'r')
    f2 = open('6/sonar_2/update_schedule.txt', 'r')
    sonar_time = {'sonar_1': '2021-07-29-09-53-53', 'sonar_2': '2021-07-29-09-53-53'}
    for data in demoloader:
        images, labels, name = data
        images, labels = images.to(device, dtype=torch.float), labels.to(device, dtype=torch.int64)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        loc = name[0].split('/')[-1].split('_')[2].split('-')
        s = name[0].split('/')[1]
        angle = (int(loc[0]) + int(loc[1]))/2
        distance = (int(loc[2]) + int(loc[3]))/ 2 / 25
        loc = sonar2pool(s, angle, distance)
        time_slot.append(name[0].split('/')[-1].split('_')[3].split('.')[0])
        prediction.append([result_map[predicted.item()], loc, s])
    sort_id = sorted(range(len(time_slot)), key=lambda k: time_slot[k])

    num_sonar = 2
    plt.ion()
    fig = plt.figure()
    fig.tight_layout()
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0.1, wspace=0.1)
    cap1 = cv2.VideoCapture("6/20210729095353.mp4")
    cap2 = cv2.VideoCapture("6/20210729103354.mp4")
    cap3 = cv2.VideoCapture("6/20210729110121.mp4")
    A = [[[] for i in range(4)] for j in range(num_sonar)]
    # each sensor will have (0) update schedule (1) point object (2) text object (3) location
    for i in range(num_sonar):
        s = ['6/sonar_1/update_schedule.txt', '6/sonar_2/update_schedule.txt'][i]
        f = open(s).readlines()
        A[i][0] = [x.strip() for x in f]

    # p1, p2, t1, t2, r1, r2 = [], [], [], [], [], []
    count = [0] * num_sonar
    ax1.set_xlim([0, 10])
    ax1.set_xlabel('location X axis/m', fontsize=20)
    ax1.set_ylim([0, 10])
    ax1.set_ylabel('location Y axis/m', fontsize=20)
    ax1.scatter([4.8, 8], [0, 0], s=100, marker='s',  c='red')
    ax1.text(4.8, 0.5, 'sonar_1', fontsize=20)
    ax1.text(8, 0.5, 'sonar_2', fontsize=20)
    ax1.set_aspect(1.25)
    for id in sort_id:
        t = time_slot[id]
        ax1.set_title('Visualization of the sonar result\n' + t)
        r = prediction[id]
        s = r[2]
        if t > "2021-07-29-11-01-23":
            t_start = datetime.strptime('2021-07-29-11-01-21', "%Y-%m-%d-%H-%M-%S")
            cap = cap3
        elif t > "2021-07-29-10-33-56":
            t_start = datetime.strptime('2021-07-29-10-33-54', "%Y-%m-%d-%H-%M-%S")
            cap = cap2
        elif t < "2021-07-29-10-24-50":
            t_start = datetime.strptime('2021-07-29-09-53-53', "%Y-%m-%d-%H-%M-%S")
            cap = cap1
        else:
            continue
        sonar_merge = []
        for i in range(num_sonar):
            if t >= A[i][0][count[i]] and len(A[i][1]) > 0:
                for j in range(len(A[i][1])):
                    if A[i][3][j]:
                        A[i][1][j].remove()
                        A[i][2][j].remove()
                A[i][1:] = [], [], []
            if i == (int(s[-1])-1):
                while (count[i] < len(A[i][0])):
                    if A[i][0][count[i]] < t:
                        count[i] = count[i] + 1
                    else:
                        break
                A[i][1].append(ax1.scatter(r[1][0], r[1][1], s=40, c='blue'))
                A[i][2].append(ax1.text(r[1][0], r[1][1], r[0] + '\n' + '(' + str(r[1][0])[:3] + ',' + str(r[1][1])[:3] + ')', fontsize=15))
                A[i][3].append([r[1][0], r[1][1], r[0]])
            if len(A[i][1]) > 0:
                sonar_merge.append(A[i][1:])
            else:
                sonar_merge.append([[], [], []])
        sonar_merge = points_merge(sonar_merge)
        for i in range(num_sonar):
            A[i][1:] = sonar_merge[i]
        seconds = (datetime.strptime(t, "%Y-%m-%d-%H-%M-%S") - t_start).total_seconds()
        i = int(cap.get(5)) * seconds
        now_time = datetime.strptime(t, "%Y-%m-%d-%H-%M-%S")
        cap.set(cv2.CAP_PROP_POS_FRAMES, i - 6)
        for j in range(12):
            ax2.cla()
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.scatter([850, 1850], [800, 800], s=100, marker='s', c='red')
            ax2.text(850, 800, 'sonar_1', fontsize=20)
            ax2.text(1850, 800, 'sonar_2', fontsize=20)
            ret, frame = cap.read()
            if ret and j % 2 == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ax2.set_title('RGB video\n' + now_time.strftime("%Y-%m-%d-%H-%M-%S"))
                ax2.imshow(frame)
                plt.pause(0.00001)