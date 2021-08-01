import torchvision.transforms as transforms
import torch.optim as optim
from models import *
from dataset import SonarDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

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
    k = 10
    kfold = KFold(n_splits=k, shuffle=True)
    EPOCH = 40
    LR = 0.01
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    Norm = transforms.Normalize((54.07982728807336, 54.04604570125005, 54.0323485177939),(15.67936276916642, 15.689930178851682, 15.699009525641682))
    demo_Norm = transforms.Normalize((49.22577382110795, 48.2819096680317, 47.88494520357551), (22.19726084193609, 21.80360678792307, 21.91796432693386))

    transform = transforms.Compose([transforms.ToTensor(), demo_Norm])
    train_transform = transform
    valid_transform = transform
    test_transform = transform

    dataset = SonarDataset(filename='demo.txt', transform=train_transform)

    if args.mode == 0:
        results = {}
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            net = LeNet((36, 18), 3).to(device)

            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
            trainloader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=train_subsampler)
            testloader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=test_subsampler)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH)
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
                    dataiter = iter(testloader)
                    correct = 0
                    total = 0
                    if epoch % 10 == 0:
                        with torch.no_grad():
                            for data in testloader:
                                images, labels = data
                                images, labels = images.to(device, dtype=torch.float), labels.to(device, dtype=torch.int64)
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                total += labels.size(0)
                                correct += (predicted == labels).sum().item()
                        acc = 100 * correct / total
                        print("Epoch {}, Training loss {:.6f}".format(epoch, loss_train))
                        print('Accuracy of the network on validate images: %d %%' % (acc))
            results[fold] = acc
            torch.save(net.state_dict(), "checkpoint/" + str(acc)[:5] + '-' + str(fold) + '.pth')
        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k} FOLDS')
        print('--------------------------------')
        sum = 0.0
        for key, value in results.items():
            print(f'Fold {key}: {value} %')
            sum += value
        print(f'Average: {sum / len(results.items())} %')
        #torch.save(net.state_dict(), "checkpoint/" + str(acc)[:5] + '.pkl')