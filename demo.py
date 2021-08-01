# simulateneously show (1) detection result with classification result (2) image at the same time
from datetime import datetime, timedelta
import torchvision.transforms as transforms
from models import *
from dataset import SonarDataset
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

def sonar2pool(angle, distance):
    if angle > 200:
        offset = (0, 5.2)
        rotate = 118.5
    else:
        offset = (0, 2)
        rotate = 105
    angle_pool = (angle - rotate)/200*np.pi
    X = offset[0] + np.sin(angle_pool)*distance
    Y = offset[1] + np.cos(angle_pool)*distance
    return [X,Y]
if __name__ == '__main__':
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    Norm = transforms.Normalize((48.83367085738062, 47.64047295347323, 47.168341320189434), (22.14060953485468, 21.77817201673869, 21.894803217573745))
    transform = transforms.Compose([transforms.ToTensor(), Norm])
    demo_transform = transform
    demo_dataset = SonarDataset(filename='demo.txt', transform=demo_transform, inferwname=True)
    demoloader = torch.utils.data.DataLoader(demo_dataset, batch_size=1, shuffle=False)
    net = LeNet((36, 18), 3).to(device)
    net.load_state_dict(torch.load("checkpoint/100.0-0.pth"))

    result_map = {0: 'Normal', 1: 'Drowning', 2: 'Submerge'}
    prediction = {}
    time_slot = []
    prediction = []
    i = 0
    for data in demoloader:
        images, labels, name = data
        images, labels = images.to(device, dtype=torch.float), labels.to(device, dtype=torch.int64)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        loc = name[0].split('/')[1].split('_')[2].split('-')
        angle = (int(loc[0]) + int(loc[1]))/2
        distance = (int(loc[2]) + int(loc[3]))/ 2 / 25
        loc = sonar2pool(angle, distance)
        time_slot.append(name[0].split('/')[1].split('_')[3].split('.')[0])
        prediction.append([result_map[predicted.item()], loc])
    sort_id = sorted(range(len(time_slot)), key=lambda k: time_slot[k])

    plt.ion()
    fig = plt.figure(figsize=(8, 6), dpi = 300)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    cap1 = cv2.VideoCapture("6/20210729095353.mp4")
    cap2 = cv2.VideoCapture("6/20210729103354.mp4")
    cap3 = cv2.VideoCapture("6/20210729110121.mp4")
    for id in sort_id:
        t = time_slot[id]
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
        ax1.cla()
        ax1.set_xlim([0, 12])
        ax1.set_ylim([0, 8])
        ax1.yaxis.tick_right()
        ax1.invert_xaxis()
        ax1.scatter([0, 0], [2, 5.2], s = 40)
        ax1.annotate('sonar_1', xy=(0, 5.2))
        ax1.annotate('sonar_2', xy=(0, 2))
        ax1.scatter([prediction[id][1][0]], [prediction[id][1][1]], s=30)
        ax1.annotate(prediction[id][0], xy=(prediction[id][1][0], prediction[id][1][1]))
        ax1.set_title(t)
        seconds = (datetime.strptime(t, "%Y-%m-%d-%H-%M-%S") - t_start).total_seconds()
        i = int(cap.get(5)) * seconds
        for j in range(-4, 4):
            ax2.cla()
            now_time = datetime.strptime(t, "%Y-%m-%d-%H-%M-%S") + timedelta(seconds=int(j / 30))
            cap.set(cv2.CAP_PROP_POS_FRAMES, i + j)
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ax2.set_title(now_time.strftime("%Y-%m-%d-%H-%M-%S"))
            ax2.imshow(frame)
            print(time.time())
            plt.pause(0.00001)