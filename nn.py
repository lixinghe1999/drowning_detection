import torchvision.transforms as transforms
import torch.optim as optim
from models import *
from dataset import SonarDataset
from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset

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
    EPOCH = 200
    LR = 0.01
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    Norm = transforms.Normalize((54.07982728807336, 54.04604570125005, 54.0323485177939),(15.67936276916642, 15.689930178851682, 15.699009525641682))
    demo_Norm = transforms.Normalize((47.76188580038861, 47.03154234737973, 46.73351142119235), (23.2561984533074, 22.776042059810912, 22.849482616025732))
    demo_add_Norm = transforms.Normalize((41.154920220478566, 39.01154961774169, 38.05637192769865), (19.250912008147463, 18.674745266913707, 18.804419542607715))
    transform_demo = transforms.Compose([transforms.ToTensor(), demo_Norm])
    transform_demo_add = transforms.Compose([transforms.ToTensor(), demo_add_Norm])
    # valid_transform = transform
    # test_transform = transform
    datasets = []
    datasets.append(SonarDataset(filename='demo.txt', transform=transform_demo))
    datasets.append(SonarDataset(filename='demo_add.txt', transform=transform_demo_add))
    # datasets.append(SonarDataset(filename='23.txt', transform=train_transform))
    dataset = ConcatDataset(datasets)
    if args.mode == 0:
        results = {}
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            net = LeNet((36, 24), 3).to(device)
            #net = LeNet((38, 14), 2).to(device)
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
            trainloader = torch.utils.data.DataLoader(dataset, batch_size=128, sampler=train_subsampler)
            testloader = torch.utils.data.DataLoader(dataset, batch_size=64, sampler=test_subsampler)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH)
            print("Start Training...")
            best_acc = 0
            for epoch in range(EPOCH):
                    loss_train = 0.0
                    for imgs, labels in trainloader:
                        #print(imgs.shape)
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
                    with torch.no_grad():
                        for data in testloader:
                            images, labels = data
                            images, labels = images.to(device, dtype=torch.float), labels.to(device, dtype=torch.int64)
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                    acc = 100 * correct / total
                    if acc > best_acc:
                        print("Epoch {}, Training loss {:.6f}".format(epoch, loss_train))
                        print('Accuracy of the network on validate images: %d %%' % (acc))
                        best_acc = acc
                        best_net_state = net.state_dict()
            results[fold] = best_acc
            torch.save(best_net_state, "checkpoint/" + str(best_acc)[:5] + '-' + str(fold) + '.pth')
        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k} FOLDS')
        print('--------------------------------')
        sum = 0.0
        for key, value in results.items():
            print(f'Fold {key}: {value} %')
            sum += value
        print(f'Average: {sum / len(results.items())} %')