from torchvision import transforms
import  torch
from torch.utils.data import Dataset
import numpy as np
import glob
import os
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
    # mean and variance
    # [41.153403795248266, 41.10783403697201, 41.126265286197004] [16.70487376238919, 16.735242169921765, 16.75909671974799]
    # [41.11250261390974, 41.13472745992071] [16.723304967395016, 16.744497445678377]
    time_dimension = 3
    count = 0
    labels = []
    images = np.zeros([51, 11, time_dimension, 1])
    for dis in ['5/', '10/', '15/']:
        if dis == '5/':
            directories = ['Noaction1A', 'Noaction2A', 'Noaction2B', 'action1A', 'action1B', 'action2A']
        elif dis == '10/':
            directories = ['Noaction1B', 'Noaction2A', 'Noaction2B', 'action1A', 'action1B', 'action2A', 'action2B']
        elif dis == '15/':
            directories = ['Noaction1A', 'action1A', 'action2A']
        directories = directories
        save_path = 'second_dataset/stack_data/'
        for d in directories:
            path = 'second_dataset/' + dis + d + '/*.npy'
            files = glob.glob(path)
            for i in range(len(files)-time_dimension+1):
                image = np.empty((51, 11, time_dimension))
                for j in range(time_dimension):
                    image[:,:,j] = np.load(files[i+j])
                np.save(save_path + str(count) + '.npy', image)
                images = np.concatenate((images, image[:, :, :, np.newaxis]), axis=3)
                if d[0]=='N':
                    l = 0
                else:
                    l = 1
                labels.append([save_path + str(count) + '.npy', l])
                count = count + 1
    random.seed(1)
    random.shuffle(labels)
    ratio = 0.8
    save_txt(labels[:int(len(labels)*ratio)], 'second_dataset/train.txt')
    save_txt(labels[int(len(labels)*ratio):], 'second_dataset/test.txt')
    print(np.shape(images))
    means, stdevs = [], []
    for i in range(time_dimension):
        pixels = images[:, :, i, :].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))
    print(means, stdevs)