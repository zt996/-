# -
课程作业测试
import torch.nn as nn
import torch.optim
import torchvision
from torch.utils.data import DataLoader
from torch.nn import  Conv2d,MaxPool2d,Flatten,Linear
train_data = torchvision.datasets.CIFAR10(root='data',transform=torchvision.transforms.ToTensor(),download=True)
train_loader = DataLoader(train_data,batch_size=64,shuffle=True,drop_last=True)

class Tudui(nn.Module):
    def __int__(self):
        super().__init__()
        self.model=nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )


    def forward(self,x):
        output=self.model(x)
        return output
tudui=Tudui()
cross=nn.CrossEntropyLoss()
optim=torch.optim.SGD(params=tudui.parameters(),lr=1e-2)
total_loss=0
for epoch in range(20):
    for data in train_loader:
        img,label=data
        output=tudui(img)
        loss_fn=cross(output,label)

        optim.zero_grad()
        loss_fn.backward()
        optim.step()


    def test(loader, model):
        correct = 0
        num = 0
        for data in loader:
            img, label = data
            output = model(img)
            correct += (output.argmax(1) == label).sum()
            num += output.size()
        return correct / num


    accuracy = test(test_loader, tudui)
print(accuracy)
