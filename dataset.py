from torchvision import transforms
import  torch
from torch.utils.data import Dataset
import numpy as np
import os
from skimage.transform import resize
import random

class SonarDataset(Dataset):
    def __init__(self, filename, repeat=1, transform=None):
        self.image_label_list = self.read_file(filename)
        self.len = len(self.image_label_list)
        self.repeat = repeat
        self.transform = transform
    def __getitem__(self, i):
        index = i % self.len
        image_path, label = self.image_label_list[index]
        img = np.load(image_path)
        label = torch.from_numpy(np.array(label))
        if self.transform:
            return self.transform(img), label
        return img, label
    def __len__(self):
        if self.repeat == None:
            data_len = 10000000
        else:
            data_len = len(self.image_label_list) * self.repeat
        return data_len
    def read_file(self, filename):
        image_label_list = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                content = line.split(' ')
                name = content[0]
                label = int(content[1])
                image_label_list.append((name, label))
        return image_label_list
def save_txt(list, fname):
    f = open(fname, 'w')
    for i in list:
        f.write(i[0] + ' ' + str(i[1]))
        f.write('\n')
    f.close()
if __name__ == '__main__':
    # label --> split --> shuffle --> calculate the mean and variance of dataset
    # mean and variance for experiment 2
    # [41.153403795248266, 41.10783403697201, 41.126265286197004] [16.70487376238919, 16.735242169921765, 16.75909671974799]
    # [41.11250261390974, 41.13472745992071] [16.723304967395016, 16.744497445678377]

    # mean and variance for experiment 2&3
    # [44.82487197607158, 44.83491829904752, 44.864822565635535] [17.306313691662208, 17.294968273741564, 17.291359324556876]

    time_dimension = 3
    count = 0
    labels = []
    test_labels = []
    images = np.zeros([60, 15, time_dimension, 1])
    directories = ['second_dataset/images/', 'third_dataset/sonar_1/images/', 'third_dataset/sonar_2/images/']
    save_path = 'whole_data/'
    for d in directories:
        files = os.listdir(d)
        for i in range(len(files)-time_dimension+1):
            image = np.empty((60, 15, time_dimension))
            flag = [f[0] for f in files[i:i+time_dimension]]
            flag = set(flag)
            if len(flag) == 1:
                for j in range(time_dimension):
                    flag = files[i+j][0]
                    path = d + files[i+j]
                    image[:, :, j] = resize(np.load(path), (60, 15))
                np.save(save_path + str(count) + '.npy', image)
                images = np.concatenate((images, image[:, :, :, np.newaxis]), axis=3)
                if files[i].split('-')[0] == 'action2':
                    test_labels.append([save_path + str(count) + '.npy', 0])
                else:
                    if 'a' in flag:
                        l = 1
                    else:
                        l = 0
                    labels.append([save_path + str(count) + '.npy', l])
                count = count + 1
    random.seed(1)
    random.shuffle(labels)
    ratio = 0.8
    save_txt(labels[:int(len(labels)*ratio)], 'train.txt')
    save_txt(labels[int(len(labels)*ratio):], 'validate.txt')
    save_txt(test_labels, 'test.txt')
    print(np.shape(images))
    means, stdevs = [], []
    for i in range(time_dimension):
        pixels = images[:, :, i, :].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))
    print(means, stdevs)