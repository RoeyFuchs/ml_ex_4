import torch
import torch.nn as nn
import torch.nn.functional as F


class Model_E(nn.Module):
    def __init__(self, image_size):
        super(Model_E, self).__init__()
        self.image_size = image_size
        self.fc1 = nn.Linear(image_size, 128)  # first layer
        self.fc2 = nn.Linear(128, 64)  # second layer
        self.fc3 = nn.Linear(64, 10)   # third layer
        self.fc4 = nn.Linear(10, 10)   # forth layer
        self.fc5 = nn.Linear(10, 10)   # fifth layer
        self.fc6 = nn.Linear(10, 10)  # output


    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        x = self.fc6(x)
        return F.log_softmax(x, dim=1)
