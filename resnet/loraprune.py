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
from peft import get_peft_model, LoraConfig, TaskType
import re


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

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove first 7 characters (i.e., 'module.')
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

# Load the state dict
state_dict = torch.load("model_weights/resnet10.pth")["net"]

new_state_dict = remove_module_prefix(state_dict)

# Load the modified state dict
my_model = ResNet50()
my_model.load_state_dict(new_state_dict)
my_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
my_model.to(device)

transform = transforms.Compose([

    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
])

forget_class_index = 3

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

filtered_indices = [i for i, (_, label) in enumerate(train_dataset) if label != forget_class_index]
filtered_train_dataset = torch.utils.data.Subset(train_dataset, filtered_indices)

trainloader = torch.utils.data.DataLoader(filtered_train_dataset, batch_size=128, shuffle=True, num_workers=2)

amount_to_prune = 0.5  # This means 50% of the channels will be pruned

# List of layers to prune; in a ResNet, these are typically the Conv2d layers
for name, module in my_model.named_modules():
    if isinstance(module, nn.Conv2d):
        # Apply random structured pruning
        prune.ln_structured(module, name='weight', amount=amount_to_prune, dim=0, n=2)
        prune.remove(module, 'weight')
        print(name)

# Define a regex pattern to match module names containing "conv1" or "conv2"
pattern = re.compile(r'.*(\.(conv1|conv2|conv3|fc))(?!.*dropout).*')

# Get all modules in the model that match the pattern
target_modules = [name for name, _ in my_model.named_modules() if pattern.match(name)]

# Use the target_modules list in your LoraConfig
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=target_modules,
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["linear","classifier"]
)

lora_resnet = get_peft_model(my_model, lora_config)
lora_resnet.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(lora_resnet.parameters(), lr=1e-3)

num_epochs = 5

for epoch in range(num_epochs):
    lora_resnet.train()  # Set the model to training mode
    running_loss = 0.0

    for images, labels in trainloader:
        # Move data to the same device as the model
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  # Zero the gradients
        outputs = lora_resnet(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters

        running_loss += loss.item()

    # Print the average loss for this epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}')

torch.save(lora_resnet.state_dict(), "model_weights/resnet10_0.5L2_lora5.pth")