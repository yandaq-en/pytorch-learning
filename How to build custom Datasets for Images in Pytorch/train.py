import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from custom_dataset import CatsAndDogsDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

in_channels = 3
num_classes = 2
learning_rate = 0.001
batch_size = 32
num_epochs = 1

# load data
dataset = CatsAndDogsDataset(csv_file="../dataset/CATDOG/labels.csv", root_dir="../dataset/CATDOG/train", transform=transforms.ToTensor())
train_set, test_set = torch.utils.data.random_split(dataset, [20000, 5000])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)


# model
model = torchvision.models.alexnet(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.classifier = nn.Linear(in_features=9216, out_features=2, bias=True)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        scores = model(data)
        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).squeeze(1)
            y = y.to(device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        acc = num_correct / num_samples
        print(f"Got {num_correct}/{num_samples} with accuracy {float(num_correct / num_samples) * 100:.2f}")

    model.train()
    return acc


print(f"Checking on the training data with device {device}")
check_accuracy(train_loader, model)
print(f"Checking on the testing data with device {device}")
check_accuracy(test_loader, model)

