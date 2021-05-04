# import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torch.nn.functional as F
import torchvision
import sys


# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyper-parameters
in_channels = 3
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# load data
train_dataset = datasets.CIFAR10(root="dataset/", train=True, transform=transforms.ToTensor(), download=False)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.CIFAR10(root="dataset/", train=False, transform=transforms.ToTensor(), download=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# load pre-train model
model = torchvision.models.vgg16(pretrained=True)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


model.avgpool = Identity()

for param in model .parameters():
    param.requires_grad = False

model.classifier = nn.Linear(512, 10)
model.to(device)


# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent
        optimizer.step()


# check accuracy
def check_accuracy(loader, model):
    if loader.dataset.train:
        print(f"Checking on the training data with device {device}")
    else:
        print(f"Checking on the testing data with device {device}")
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        acc = num_correct / num_samples
        print(f"Got {num_correct}/{num_samples} with accuracy {float(num_correct / num_samples) * 100:.2f}")

    model.train()
    return acc


check_accuracy(test_loader, model)
