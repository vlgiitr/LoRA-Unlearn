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
from torchvision.models.vision_transformer import MLPBlock

mixup = v2.MixUp(alpha=0.2, num_classes=10)

warmup_try=10000

# Taken from https://github.com/lucidrains/vit-pytorch, likely ported from https://github.com/google-research/big_vision/
def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

        # Fix init discrepancy between nn.MultiheadAttention and that of big_vision
        bound = math.sqrt(3 / hidden_dim)
        nn.init.uniform_(self.self_attention.in_proj_weight, -bound, bound)
        nn.init.uniform_(self.self_attention.out_proj.weight, -bound, bound)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        return self.ln(self.layers(self.dropout(input)))


class SimpleVisionTransformer(nn.Module):
    """Vision Transformer modified per https://arxiv.org/abs/2205.01580."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 10,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        self.conv_proj = nn.Conv2d(
            in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
        )

        h = w = image_size // patch_size
        seq_length = h * w
        self.register_buffer("pos_embedding", posemb_sincos_2d(h=h, w=w, dim=hidden_dim))

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            # constant is stddev of standard normal truncated to (-2, 2)
            std = math.sqrt(1 / fan_in) / .87962566103423978
            nn.init.trunc_normal_(self.conv_proj.weight, std=std, a=-2 * std, b=2 * std)
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)

        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)



        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        x = x + self.pos_embedding
        x = self.encoder(x)
        x = x.mean(dim = 1)
        x = self.heads(x)

        return x

def weight_decay_param(n, p):
    return p.ndim >= 2 and n.endswith('weight')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create model
model = SimpleVisionTransformer(
    image_size=32,
    patch_size=4,
    num_layers=12,
    num_heads=6,
    hidden_dim=384,
    mlp_dim=1536,
).to(device)
wd_params = [p for n, p in model.named_parameters() if weight_decay_param(n, p) and p.requires_grad]
non_wd_params = [p for n, p in model.named_parameters() if not weight_decay_param(n, p) and p.requires_grad]

model9 = SimpleVisionTransformer(
    image_size=32,
    patch_size=4,
    num_layers=12,
    num_heads=6,
    hidden_dim=384,
    mlp_dim=1536,
).to(device)

my_model = model
my_model.load_state_dict(torch.load(f"model_weights/vitbase10_prune_lora10_headatt.pth"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
my_model.to(device)

# Set up transformations for the dataset (e.g., normalization)
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
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=2)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
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