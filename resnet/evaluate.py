import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torchvision.models as models
from datasets import load_dataset, Dataset
from torch.optim import Adam
import os
import shutil
import time
import math
from enum import Enum
from functools import partial
from collections import OrderedDict
import torch.utils.data
import torchvision.datasets as datasets
from torchvision.transforms import v2
import torchvision.transforms as transforms
from typing import Any, Dict, Union, Type, Callable, Optional, List

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


# Load the state dict
state_dict = torch.load("model_weights/resnet10_finetune5.pth")

# Load the modified state dict
my_model = ResNet50()
my_model.load_state_dict(state_dict)
my_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
my_model.to(device)

transform = transforms.Compose([

    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
])

forget_class_index = 3

def evaluate_model_accuracy(model, valloader, device):
    model.eval()  # Set the model to evaluation mode
    correct_count = 0
    total_count = 0

    with torch.no_grad():
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # Get the predicted class (assuming it's a classification task)
            _, predicted = torch.max(outputs.data, 1)

            total_count += labels.size(0)
            correct_count += (predicted == labels).sum().item()

    accuracy = correct_count / total_count
    return accuracy

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

filtered_indices = [i for i, (_, label) in enumerate(train_dataset) if label != forget_class_index]
filtered_train_dataset = torch.utils.data.Subset(train_dataset, filtered_indices)

filtered_indices = [i for i, (_, label) in enumerate(test_dataset) if label != forget_class_index]
filtered_test_dataset = torch.utils.data.Subset(test_dataset, filtered_indices)

forget_indices = [i for i, (_, label) in enumerate(train_dataset) if label == forget_class_index]
forget_train_dataset = torch.utils.data.Subset(train_dataset, forget_indices)

forget_indices = [i for i, (_, label) in enumerate(test_dataset) if label == forget_class_index]
forget_test_dataset = torch.utils.data.Subset(test_dataset, forget_indices)

trainloader = torch.utils.data.DataLoader(filtered_train_dataset, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(filtered_test_dataset, batch_size=128, shuffle=True, num_workers=2)
org_trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
forget_trainloader = torch.utils.data.DataLoader(forget_train_dataset, batch_size=128, shuffle=True, num_workers=2)
forget_testloader = torch.utils.data.DataLoader(forget_test_dataset, batch_size=128, shuffle=False, num_workers=2)

accuracy = evaluate_model_accuracy(my_model, forget_trainloader, device)
print(f"Unlearning Accuracy : {accuracy:.2%}")

accuracy = evaluate_model_accuracy(my_model, trainloader, device)
print(f"Remaining Accuracy : {accuracy:.2%}")

accuracy = evaluate_model_accuracy(my_model, testloader, device)
print(f"Testing Accuracy : {accuracy:.2%}")

def compute_losses(net, loader):
    """Auxiliary function to compute per-sample losses"""

    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        logits = net(inputs)
        losses = criterion(logits, targets).numpy(force=True)
        for l in losses:
            all_losses.append(l)

    return np.array(all_losses)

resnet10_losses = compute_losses(my_model, trainloader)
resnet10_test_losses = compute_losses(my_model, testloader)
np.random.shuffle(resnet10_test_losses)
resnet10_test_losses = resnet10_test_losses[:5000]

mia_filtered_train_dataset = np.concatenate((resnet10_losses[:1000], resnet10_losses[5000:6000] , resnet10_losses[10000:11000] , resnet10_losses[15000:16000] , resnet10_losses[20000:21000] , resnet10_losses[25000:26000] , resnet10_losses[30000:31000] , resnet10_losses[35000:36000] , resnet10_losses[40000:41000]))
mia_filtered_test_dataset = resnet10_test_losses

forget_losses = compute_losses(my_model,forget_trainloader)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

samples_mia = np.concatenate((mia_filtered_train_dataset, mia_filtered_test_dataset)).reshape((-1, 1))
labels_mia = [0] * len(mia_filtered_train_dataset) + [1] * len(mia_filtered_test_dataset)

forget_losses = forget_losses.reshape((-1,1))
forget_labels = [1] * len(forget_losses)

# Separate features and labels for the datasets
X_train, y_train = samples_mia, labels_mia
X_forget, y_forget = forget_losses, forget_labels

# Train Logistic Regression model on the filtered training set
mia_model = LogisticRegression()
mia_model.fit(X_train, y_train)

# Make predictions on the forget dataset (Df)
y_pred_forget = mia_model.predict(X_forget)

# Calculate confusion matrix for the forget dataset
tn = np.count_nonzero(y_pred_forget == 1)

# Calculate MIA-Efficacy
mia_efficacy = tn / len(forget_train_dataset)
print(f"MIA-Efficacy: {mia_efficacy}")

plt.title("Losses on train and forget set")
plt.hist(resnet10_test_losses, density=True, alpha=0.6, bins=50, label="Test Losses set")
plt.hist(forget_losses, density=True, alpha=0.4, bins=50, label="Forget Losses set")
plt.xlabel("Loss", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.xlim((0, np.max(forget_losses)+2))
plt.yscale("log")
plt.legend(frameon=False, fontsize=14)
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.show()