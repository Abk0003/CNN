import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
from torch.utils.data import  DataLoader
from torchvision import transforms, datasets as Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

cifar_train = Dataset.CIFAR10(root="../data", train = True, transform = transforms.ToTensor(),download=True)
cifar_test = Dataset.CIFAR10(root="../data", train = False, transform = transforms.ToTensor(),download=True)

train_loader = DataLoader(dataset=cifar_train, batch_size=128,shuffle=True)
test_loader = DataLoader(dataset=cifar_test, batch_size=128,shuffle=True)
images, labels = next(iter(train_loader))
print(images.shape)
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1,padding =1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3,stride=1,padding =1),

            nn.Flatten(),

            nn.Linear(in_features=1024*4*4,out_features=256),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=64, out_features=10),
        )
    def forward(self,x):
        x = self.classifier(x)
        return x

model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(50):
    lossT = 0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        lossT += loss.item()
        total += labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    loss = lossT/len(train_loader)
    acc = correct / total
    print(epoch, loss, acc)

model.eval()  # important: disables dropout/batchnorm training behavior

correct = 0
total = 0
test_loss = 0

criterion = nn.CrossEntropyLoss()

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item()

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = test_loss / len(test_loader)
    accuracy = correct / total

    print("Test Loss:", avg_loss)
    print("Test Accuracy:", accuracy)





