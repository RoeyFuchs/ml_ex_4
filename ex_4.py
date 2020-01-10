import torch
from torchvision import transforms
from torchvision import datasets
from Model_A import Model_A
from Model_B import Model_B
from Model_C import Model_C
from Model_D import Model_D
from Model_E import Model_E



import matplotlib.pyplot as plt
import numpy as np


import torch.optim as optim
import torch.nn.functional as F


def plotDet(loss_train_set,acc_train_set,loss_valid_set,acc_valid_set ):
    plt.figure()
    plt.title('Avg. loss as function of epoch')
    plt.ylabel('Avg. loss')
    plt.xlabel('epoch #')
    plt.plot(loss_train_set, label='training set')
    plt.plot(loss_valid_set, label='validation set')
    plt.legend()
    plt.show()
    plt.figure()
    plt.title('Accuracy as function of epoch')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch #')
    plt.plot(acc_train_set, label='training set')
    plt.plot(acc_valid_set, label='validation set')
    plt.legend()
    plt.show()


def train(epoch, model):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()


def test(model, check_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in check_loader:
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).cpu().sum()

    test_loss /= len(check_loader.dataset)
    acc = (100. * correct / len(check_loader.dataset))
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(check_loader.dataset),
        acc))

    return test_loss, acc

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0,), (1,))])
batchSize = 64
train_data_before_split = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)

train_size = int(len(train_data_before_split.train_data) * 0.8)
valid_size = len(train_data_before_split.train_data) - train_size


data_for_train, data_for_valid = torch.utils.data.random_split(train_data_before_split, [train_size, valid_size])

train_loader = torch.utils.data.DataLoader(data_for_train, batch_size=batchSize, shuffle=True)
validation_set = torch.utils.data.DataLoader(data_for_valid, batch_size=batchSize, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./data', train=False, download=True, transform=transform),
    batch_size=batchSize, shuffle=True
)

model = Model_E(image_size=28 * 28)
lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)

epohcs_number = 10

loss_train_set = np.zeros(epohcs_number)
acc_train_set = np.zeros(epohcs_number)
loss_valid_set = np.zeros(epohcs_number)
acc_valid_set = np.zeros(epohcs_number)
for epoch in range(epohcs_number):
    train(epoch, model)
    loss_train_set[epoch], acc_train_set[epoch] = test(model, train_loader)
    loss_valid_set[epoch], acc_valid_set[epoch] = test(model, validation_set)

plotDet(loss_train_set, acc_train_set, loss_valid_set, acc_valid_set)
