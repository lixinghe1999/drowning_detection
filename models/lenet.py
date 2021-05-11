'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, shape):
        super(LeNet, self).__init__()
        kernel = (3, 3)
        num_fc1, num_fc2 = shape
        for i in range(2):
            num_fc1 = int((num_fc1 - kernel[0] + 1) / 2)
            num_fc2 = int((num_fc2 - kernel[1] + 1) / 2)
        self.conv1 = nn.Conv2d(4, 8, kernel)
        self.conv2 = nn.Conv2d(8, 16, kernel)
        self.fc1   = nn.Linear(16 * num_fc1* num_fc2, 120)
        self.fc2   = nn.Linear(120, 20)
        self.fc3   = nn.Linear(20, 2)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
