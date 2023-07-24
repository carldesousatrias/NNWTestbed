# vgg16
import torch
import torch.nn.functional as F
import math

from utils import *


def get_image(name):
    """
    :param name: file (including the path) of an image
    :return: a numpy of this image
    """
    image = Image.open(name)
    return np.array(image)

class ROUH_ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size()[0], -1)

class ROUH_LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.flatten = Flatten()
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.flatten(out)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        feat = self.relu1(out)
        out = self.fc2(feat)
        return F.softmax(out, dim=1), feat


class ROUH_fullyconnected(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc=nn.Linear(3072,512)
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(0.45)
        self.fc1 = nn.Linear(512, 512)
        self.relu1 = nn.ReLU()
        self.dp1 = nn.Dropout(0.45)
        self.fc2 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()
        self.dp2 = nn.Dropout(0.45)
        self.fc3 = nn.Linear(512, num_classes)


    def forward(self, x):
        x = x.view((x.size()[0], -1))
        out = self.fc(x)
        out = self.relu(out)
        out = self.dp(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.dp1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        feat = self.dp2(out)
        out = self.fc3(out)
        return F.softmax(out, dim=1), feat


if __name__=='__main__':
    x=torch.randn(1,1,28,28)
    net=ToyNet()
    y=net(x)