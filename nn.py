import torchvision.transforms as transforms
import torch.optim as optim
from models import *
from dataset import SonarDataset
import matplotlib.pyplot as plt
import time
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='analyse experiment data')
    parser.add_argument('--mode', action="store", required=False, type=int, default=0, help="0-training and validating, 1-testing")
    args = parser.parse_args()
    if args.mode == 0:
        device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        #device = torch.device('cpu')
        print(f"Training on device {device}.")
        net = LeNet().to(device)
        #net = ResNet(ResidualBlock).to(device)
        #net = MobileNetV2().to(device)
        #net = ResNet18().to(device)
        """
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint)
        """
        EPOCH = 200
        LR = 0.01
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        #Norm = transforms.Normalize((41.11250261390974, 41.13472745992071), (16.723304967395016, 16.744497445678377))
        #Norm = transforms.Normalize((41.153403795248266, 41.10783403697201, 41.126265286197004),(16.70487376238919, 16.735242169921765, 16.75909671974799))
        Norm = transforms.Normalize((44.82487197607158, 44.83491829904752, 44.864822565635535), (17.306313691662208, 17.294968273741564, 17.291359324556876))
        train_transform = transforms.Compose([transforms.ToTensor(), Norm])
        valid_transform = transforms.Compose([transforms.ToTensor(), Norm])
        train_dataset = SonarDataset(filename='train.txt',  transform=train_transform)
        valid_dataset = SonarDataset(filename='validate.txt',  transform=valid_transform)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False)
        validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=128, shuffle=False)

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
                    print("Epoch {}, Training loss {:.6f}".format(epoch, loss_train / len(trainloader)))
                    print('Accuracy of the network on validate images: %d %%' % (acc) )
        torch.save(net.state_dict(), "checkpoint/" + str(acc)[:5] + '.pkl')
        print("Done Training!")
    else:
        device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        # device = torch.device('cpu')
        net = LeNet().to(device)
        net.load_state_dict(torch.load("checkpoint/94.01.pkl"))
        Norm = transforms.Normalize((44.82487197607158, 44.83491829904752, 44.864822565635535), (17.306313691662208, 17.294968273741564, 17.291359324556876))
        test_transform = transforms.Compose([transforms.ToTensor(), Norm])
        test_dataset = SonarDataset(filename='test.txt', transform=test_transform)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
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
        plt.plot(result)
        plt.show()


