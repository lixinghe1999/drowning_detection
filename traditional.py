from dataset import  SonarDataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
Norm = ((41.153403795248266, 41.10783403697201, 41.126265286197004),(16.70487376238919, 16.735242169921765, 16.75909671974799))
train_dataset = SonarDataset(filename='second_dataset/train.txt')
test_dataset = SonarDataset(filename='second_dataset/test.txt')
diff = [[], []]
j = 0
correct = 0
for img, label in train_dataset:
    a = 0
    #for i in range(np.shape(img)[2]):
        #img[:,:,i] /= np.max(img[:,:,i].ravel())
    a += np.mean(abs(img[:, :, 1] - img[:, :, 0]).ravel())
    a += np.mean(abs(img[:, :, 2] - img[:, :, 1]).ravel())
    a = a/2
    diff[label].append(a)
    # if a >= 40.8 and label == 1:
    #     correct +=1
    # elif a<40.8 and label ==0:
    #     correct +=1
    j = j+1
    if (j>500):
        break
print(np.mean(diff[0]), np.mean(diff[1]))
print(correct/j)