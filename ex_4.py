import torch
from torchvision import transforms
from torchvision import datasets
from Model_A import Model_A
from Model_B import Model_B


import torch.optim as optim
import torch.nn.functional as F


def train(epoch, model):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()


def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0,), (1,))])
batchSize = 64
train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./data', train=True, download=True, transform=transform),
    batch_size=batchSize, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./data', train=False, download=True, transform=transform),
    batch_size=batchSize, shuffle=True
)

model = Model_B(image_size=28 * 28)
lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(10):
    train(epoch, model)
    test(model)