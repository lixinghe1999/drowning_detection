from torchvision import transforms
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from skimage.transform import resize
import matplotlib.pyplot as plt
import random
import shutil

class SonarDataset(Dataset):
    def __init__(self, filename, repeat=1, transform=None, inferwname=False):
        self.image_label_list = self.read_file(filename)
        self.len = len(self.image_label_list)
        self.repeat = repeat
        self.transform = transform
        self.inferwname = inferwname
    def __getitem__(self, i):
        index = i % self.len
        image_path, label = self.image_label_list[index]
        img = np.load(image_path)
        label = torch.from_numpy(np.array(label))
        if self.inferwname and self.transform:
            return self.transform(img), label, image_path
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
    import argparse

    parser = argparse.ArgumentParser(description='Control the sonar')
    parser.add_argument('--mode', action="store", required=False, type=int, default=3, help="Preprocess or real process")
    args = parser.parse_args()
    # label --> split --> shuffle --> calculate the mean and variance of dataset
    # mean and variance for experiment 2
    # [41.153403795248266, 41.10783403697201, 41.126265286197004] [16.70487376238919, 16.735242169921765, 16.75909671974799]
    # [41.11250261390974, 41.13472745992071] [16.723304967395016, 16.744497445678377]

    # mean and variance for experiment 2&3
    # [44.82487197607158, 44.83491829904752, 44.864822565635535] [17.306313691662208, 17.294968273741564, 17.291359324556876]
    # [48.28077004319593, 48.29275944397162, 48.3016474874335] [18.304855477888243, 18.288180591815973, 18.288212328981537]
    # [52.599049512970446, 52.580069378570286, 52.56049022923118] [15.84267112429285, 15.855886602198726, 15.866821187867181]

    if args.mode == 0:
        #directories = ['2/images/', '3/sonar_1/images/', '3/sonar_2/images/']
        directories = ['6/sonar_1/images/', '6/sonar_2/images/', '7/images/']
        Max_shape = (0, 0)
        for d in directories:
            files = os.listdir(d)
            for i in range(len(files)):
                path = d + files[i]
                shape = np.load(path).shape
                plt.scatter(shape[0], shape[1])
                Max_shape = np.max([shape[:2], Max_shape], axis=0)
        print(Max_shape)
        plt.show()
    elif args.mode == 1:
        directories = ['2/images/', '3/sonar_1/images/', '3/sonar_2/images/']
        Max_shape = (38, 14)
        time_dimension = 3
        count = 0
        labels = []
        test_labels = []
        images = np.zeros([Max_shape[0], Max_shape[1], time_dimension, 1])
        save_path = 'whole_data/'
        for d in directories:
            files = os.listdir(d)
            for i in range(len(files)-time_dimension+1):
                image = np.empty((Max_shape[0], Max_shape[1], time_dimension))
                flag = [f[0] for f in files[i:i+time_dimension]]
                flag = set(flag)
                if len(flag) == 1:
                    for j in range(time_dimension):
                        flag = files[i+j][0]
                        path = d + files[i+j]
                        image[:, :, j] = resize(np.load(path), (Max_shape[0], Max_shape[1]))
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
        save_txt(labels, '23.txt')
        save_txt(test_labels, 'test.txt')
        print(np.shape(images))
        means, stdevs = [], []
        for i in range(time_dimension):
            pixels = images[:, :, i, :].ravel()  # 拉成一行
            means.append(np.mean(pixels))
            stdevs.append(np.std(pixels))
        print(means, stdevs)
    elif args.mode == 2:
        count = 0
        Max_shape = (36, 24)
        time_dimension = 3
        labels = []
        images = np.zeros([Max_shape[0], Max_shape[1], time_dimension, 1])
        for s in ['sonar_1/', 'sonar_2/']:
            shutil.rmtree('6/' + s + 'demo_data/')
            os.mkdir('6/' + s + 'demo_data/')
            d = '6/' + s + 'images/'
            save_path = '6/' + s + 'demo_data/'
            files = os.listdir(d)
            for file in files:
                image = np.load(d + file)
                for i in range(0, image.shape[2]-time_dimension + 1):
                    a = resize(image[:, :, i: i+time_dimension], (Max_shape[0], Max_shape[1], time_dimension))
                    np.save(save_path + str(count) + '_' + file, a)
                    images = np.concatenate((images, a[:, :, :, np.newaxis]), axis=3)
                    labels.append([save_path + str(count) + '_' + file, file.split('_')[0]])
                    count = count + 1
        save_txt(labels, 'demo.txt')
        print(np.shape(images))
        means, stdevs = [], []
        for i in range(time_dimension):
            pixels = images[:, :, i, :].ravel()  # 拉成一行
            means.append(np.mean(pixels))
            stdevs.append(np.std(pixels))
        print(means, stdevs)
    elif args.mode == 3:
        count = 0
        Max_shape = (36, 24)
        time_dimension = 3
        labels = []
        images = np.zeros([Max_shape[0], Max_shape[1], time_dimension, 1])
        shutil.rmtree('7/demo_data/')
        os.mkdir('7/demo_data/')
        d = '7/images/'
        save_path = '7/demo_data/'
        files = os.listdir(d)
        for file in files:
            # if file.split('_')[0] == '3':
            #     image = np.load(d + file)
            #     for i in range(0, image.shape[2] - time_dimension + 1):
            #         a = resize(image[:, :, i: i + time_dimension], (Max_shape[0], Max_shape[1], time_dimension))
            #         a /= a.max()
            #         plt.subplot(1,3, 1)
            #         plt.imshow(a[:,:,0])
            #         plt.subplot(1, 3, 2)
            #         plt.imshow(a[:,:,1])
            #         plt.subplot(1, 3, 3)
            #         plt.imshow(a[:,:,2])
            #         plt.show()
            #     continue
            # else:
            image = np.load(d + file)
            for i in range(0, image.shape[2] - time_dimension + 1):
                a = resize(image[:, :, i: i + time_dimension], (Max_shape[0], Max_shape[1], time_dimension))
                np.save(save_path + str(count) + '_' + file, a)
                images = np.concatenate((images, a[:, :, :, np.newaxis]), axis=3)
                labels.append([save_path + str(count) + '_' + file, file.split('_')[0]])
                count = count + 1
        save_txt(labels, 'demo_add.txt')
        print(np.shape(images))
        means, stdevs = [], []
        for i in range(time_dimension):
            pixels = images[:, :, i, :].ravel()  # 拉成一行
            means.append(np.mean(pixels))
            stdevs.append(np.std(pixels))
        print(means, stdevs)
