import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torchsummary import summary
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 32

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        self.linear = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out,out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def project1_model():
    return ResNet(BasicBlock, [2,2,2,2])

def test():
    net=project1_model()
    y=net(torch.randn(1,3,32,32))
    print(y.size())


trainingdata = torchvision.datasets.CIFAR10('./CIFAR10/',train=True,download=True,transform=torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=2),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))]))
testdata = torchvision.datasets.CIFAR10('./CIFAR10/',train=False,download=True,transform=torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=2),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))]))

trainDataLoader = torch.utils.data.DataLoader(trainingdata,batch_size=400,shuffle=True)
testDataLoader = torch.utils.data.DataLoader(testdata,batch_size=800,shuffle=False)

classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

net=project1_model().cuda()

Loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0005)

train_loss_history = []
test_loss_history = []
summary(net,(3,32,32))

bestacc=float('-inf')

for epoch in range(150):
  train_loss=0.0
  test_loss=0.0
  correct_train=0
  correct_test=0
  total_train=0
  total_test=0

  for i, data in enumerate(trainDataLoader):
    images, labels = data
    images = images.cuda()
    labels = labels.cuda()
    optimizer.zero_grad()
    predicted_output = net(images)
    fit = Loss(predicted_output,labels)
    fit.backward()
    optimizer.step()
    train_loss += fit.item()

  train_loss = train_loss/len(trainDataLoader)
  train_loss_history.append(train_loss)
  total_train+=labels.size(0)
  l,predicted=predicted_output.max(1)
  correct_train+=predicted.eq(labels).sum().item()
  trainacc=100.*correct_train/total_train

  for i, data in enumerate(testDataLoader):
    with torch.no_grad():
      images, labels = data
      images = images.cuda()
      labels = labels.cuda()
      predicted_output = net(images)
      fit = Loss(predicted_output,labels)
      test_loss += fit.item()
  
  
  test_loss = test_loss/len(testDataLoader)
  test_loss_history.append(test_loss)
  l,predicted=predicted_output.max(1)
  total_test+=labels.size(0)
  correct_test+=predicted.eq(labels).sum().item()
  testacc=100.*correct_test/total_test
  print('Epoch %s, Train loss %s, Test loss %s, Train_Accuracy %s, Test_Accuracy %s'%(epoch, train_loss, test_loss,trainacc,testacc))

  if bestacc<testacc:
      bestacc=testacc
      model_path = './project1_model.pt'
      torch.save(net.state_dict(), model_path)

plt.plot(range(150),train_loss_history,'-',linewidth=3,label='Train error')
plt.plot(range(150),test_loss_history,'-',linewidth=3,label='Test error')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid(True)
plt.legend()
#plt.show()
plt.savefig("plot.png")

# model_path = './project1_model.pt'
# torch.save(net.state_dict(), model_path)
# summary(net,(3,32,32))