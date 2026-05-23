import torch.nn as nn
import torch
import torch.nn.functional as F
from sympy import false
from torch.utils.data import  DataLoader
from torchvision import transforms, datasets as Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using  device: {device}")

cifartrain = Dataset.CIFAR100(root="./data", train=True, download = True, transform=transforms.Compose([

    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(
    brightness=0.2,
    contrast=0.2,
    saturation=0.2),
    transforms.AutoAugment(
        policy=transforms.AutoAugmentPolicy.CIFAR10
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5071, 0.4867, 0.4408),
        (0.2675, 0.2565, 0.2761)
    ),
    transforms.RandomErasing(p=0.25, scale = (0.02,0.2),ratio = (0.3,3.3),value='random')
]))
cifartest = Dataset.CIFAR100(root="./data", train=False, download = True, transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
]))

trainloader = DataLoader(cifartrain, batch_size=128, shuffle=True)
testloader = DataLoader(cifartest, batch_size=128, shuffle=True)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,stride = stride,padding=1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels or stride != 1 :
           self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
           )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + identity
        x = F.relu(x)
        return x

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.classifier = nn.Sequential(
            ResidualBlock(3,64,1),
            ResidualBlock(64,64,1),
            ResidualBlock(64,64,1),
            ResidualBlock(64,128,2),
            ResidualBlock(128,128,1),
            ResidualBlock(128,128,1),
            ResidualBlock(128,256,2),
            ResidualBlock(256,256,1),
            ResidualBlock(256,256,1),
            ResidualBlock(256,512,2),
            ResidualBlock(512,512,1),

            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),

            nn.Linear(512,100),
        )
    def forward(self,x):
        x = self.classifier(x)
        return x

Model = Model().to(device)
optimizer = torch.optim.AdamW(Model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
schedular = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0 = 100)

Model.train()
for epoch in range(100):
    lossT = 0
    correct = 0
    total = 0
    for images,labels in trainloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = Model(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lossT += loss.item()
        pred =outputs.argmax(dim=1)
        correct += (pred==labels).sum().item()
        total += labels.size(0)
    schedular.step()
    print(f"Epoch {epoch+1}, Loss: {lossT/len(trainloader)}, Accuracy: {correct/total*100:.2f}%")
    Model.eval()

    test_correct = 0
    test_total = 0
    test_loss = 0

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = Model(images)

            loss = criterion(outputs, labels)

            test_loss += loss.item()

            pred = outputs.argmax(dim=1)

            test_correct += (pred == labels).sum().item()
            test_total += labels.size(0)

    print(
        f"Test Loss: {test_loss / len(testloader)}, "
        f"Test Accuracy: {test_correct / test_total * 100:.2f}%"
    )

    Model.train()
