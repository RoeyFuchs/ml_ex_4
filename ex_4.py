import sys
import torch
from torchvision import transforms
from torchvision import datasets
from Model_C import Model_C
import numpy as np
import torch.optim as optim
import torch.nn.functional as F


'''the full code (with train-validetion-test and plots) - in ex_4_plot.py'''

# train the model
def train(epoch, model):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels, reduction='sum')
        loss.backward()
        optimizer.step()


# predict with the model
def predict(model, input_loader):
    model.eval()
    ret = np.empty(shape=0)
    for data in input_loader:
        output = model(data)
        pred = output.max(1, keepdim=True)[1]
        pred = torch.Tensor.cpu(pred).detach().numpy()[:, :]
        ret = np.append(ret, pred)
    return ret


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0,), (1,)),
                                ])  # normalize - zero mean and 1 std
batchSize = 64
train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./data', train=True, download=True, transform=transform), batch_size=batchSize, shuffle=True)
# create the model for the run, my best - model C
model = Model_C(image_size=28 * 28)
lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)
epohcs_number = 10

for epoch in range(epohcs_number):
    train(epoch, model)

# predict from a file
if (len(sys.argv) > 1):
    fileName = sys.argv[1]
    train_x = np.loadtxt(fileName) / 255  # divided by 255 as the traning set
    x = transform(train_x)[0].float()
    y = predict(model, x)
    # crate predicts file
    y = '\n'.join(str(int(x)) for x in y)
    f = open("test_y", "w")
    f.write(y)
    f.close()
