import torchvision.transforms as transforms
import torch.optim as optim
from models import *
from dataset import SonarDataset
import matplotlib.pyplot as plt
import numpy as np
import time
class delta_transform(object):
    def __call__(self, img):
        c, h, w = img.shape
        for i in range(1, c):
            img[-i, :, :] = img[-i, :, :] - img[-i-1, :, :]
        return img
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='analyse experiment data')
    parser.add_argument('--mode', action="store", required=False, type=int, default=0, help="0-training and validating, 1-testing")
    args = parser.parse_args()

    EPOCH = 200
    LR = 0.01
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    # device = torch.device('cpu')
    net = LeNet((45, 15)).to(device)
    # net.load_state_dict(torch.load("checkpoint/81.57.pkl"))
    Norm = transforms.Normalize((52.819049512970446, 52.800069378570286, 52.79049022923118), (15.84267112429285, 15.855886602198726, 15.866821187867181))
    #Norm = transforms.Normalize((52.489604005663956, 52.453554835700686, 52.448508964688735, 52.44205857533522), (15.804601841790076, 15.799356294654816, 15.811369122906537, 15.818706894728548))
    transform = transforms.Compose([transforms.ToTensor(), Norm])
    train_transform = transform
    valid_transform = transform
    test_transform = transform

    train_dataset = SonarDataset(filename='train.txt', transform=train_transform)
    valid_dataset = SonarDataset(filename='validate.txt', transform=valid_transform)
    test_dataset = SonarDataset(filename='test.txt', transform=test_transform)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=256, shuffle=False)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH)
    if args.mode == 0:

        loss_plt = []
        acc_plt = []
        print("Start Training...")
        for epoch in range(EPOCH):
                loss_train = 0.0
                for imgs, labels in trainloader:
                    imgs = imgs.to(device=device, dtype = torch.float)
                    labels = labels.to(device=device, dtype = torch.int64)
                    outputs = net(imgs)
                    loss = criterion(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    loss_train += loss.item()
                loss_train = loss_train / len(trainloader)
                dataiter = iter(validloader)
                correct = 0
                total = 0
                if epoch % 5 ==0:
                    with torch.no_grad():
                        for data in validloader:
                            images, labels = data
                            images, labels = images.to(device, dtype=torch.float), labels.to(device, dtype=torch.int64)
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                    acc = 100 * correct / total
                    loss_plt.append(loss_train)
                    acc_plt.append(acc)
                    print("Epoch {}, Training loss {:.6f}".format(epoch, loss_train))
                    print('Accuracy of the network on validate images: %d %%' % (acc) )
        torch.save(net.state_dict(), "checkpoint/" + str(acc)[:5] + '.pkl')
        print("Done Training!")
        plt.subplot(1, 2, 1)
        plt.plot(loss_plt)
        plt.subplot(1, 2, 2)
        plt.plot(acc_plt)
        plt.show()
    else:
        net.load_state_dict(torch.load("checkpoint/90.64.pkl"))
        result = []
        with torch.no_grad():
            t_start = time.time()
            for data in testloader:
                images, labels = data
                images, labels = images.to(device, dtype=torch.float), labels.to(device, dtype=torch.int64)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                result.append(predicted)
            print((time.time() - t_start) / len(testloader))
        #result = np.convolve(result, np.ones(5) / 5, mode="same")
        plt.plot(result)
        plt.show()


