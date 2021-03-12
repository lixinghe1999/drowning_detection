import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import os
from models import *

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")
#net = ResNet(ResidualBlock).to(device)
net = MobileNetV2().to(device)
#net = ResNet18().to(device)
"""
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/ckpt.pth')
net.load_state_dict(checkpoint)
"""
EPOCH = 200
LR = 0.1
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
cifar_train = torchvision.datasets.CIFAR10(root='./cifar', train=True, download=True, transform=transform_train)
cifar_test = torchvision.datasets.CIFAR10(root='./cifar', train=False, transform=transform_test)
trainloader = torch.utils.data.DataLoader(cifar_train, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(cifar_test, batch_size=128, shuffle=True)

print("Start Training...")
for epoch in range(EPOCH):
        loss_train = 0.0
        for imgs, labels in trainloader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            outputs = net(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_train += loss.item()
        print("Epoch {}, Training loss {:.6f}".format(epoch, loss_train / len(trainloader)))

print("Done Training!")

dataiter = iter(testloader)
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
acc = 100 * correct / total
print('Accuracy of the network on the 10000 test images: %d %%' % (acc))
print('Saving..')
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
torch.save(net.state_dict(), './checkpoint/%f.pth' % (acc))