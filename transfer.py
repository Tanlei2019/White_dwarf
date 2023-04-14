
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

def read_data(flux, scls, batch_size = 100):
    fluxTR, fluxTE, clsTR, clsTE = train_test_split(flux, scls, test_size=0.3)
    Xtrain1 = torch.from_numpy(fluxTR)  # numpy 转成 torch 类型
    Xtest1 = torch.from_numpy(fluxTE)
    ytrain1 = torch.from_numpy(clsTR)
    ytest1 = torch.from_numpy(clsTE)
    torch_dataset_train = Data.TensorDataset(Xtrain1, ytrain1)
    torch_dataset_test = Data.TensorDataset(Xtest1, ytest1)
    data_loader_train = torch.utils.data.DataLoader(dataset=torch_dataset_train, batch_size=batch_size, shuffle=True)
    data_loader_test = torch.utils.data.DataLoader(dataset=torch_dataset_test, batch_size=batch_size, shuffle=True)
    return data_loader_train, data_loader_test, clsTE.shape[0], clsTR.shape[0], batch_size

class CNN_Model1(torch.nn.Module):
    def __init__(self):
        super(CNN_Model1, self).__init__()
        self.conv0 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=(1, 1), stride=1),
            torch.nn.BatchNorm2d(10),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(stride=1, kernel_size=(1, 2)))
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 10, kernel_size=(1, 2), stride=1),
            torch.nn.BatchNorm2d(10),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=(1, 2)))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=(1, 3), stride=1),
            torch.nn.BatchNorm2d(20),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(stride=1, kernel_size=(1, 2)))
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(20, 30, kernel_size=(1, 4), stride=1),
            torch.nn.BatchNorm2d(30),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=(1, 2)))
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(30, 40, kernel_size=(1, 5), stride=1),
            torch.nn.BatchNorm2d(40),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(stride=1, kernel_size=(1, 2)))
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(40, 50, kernel_size=(1, 7), stride=1),
            torch.nn.BatchNorm2d(50),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=(1, 2)))
        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(50, 60, kernel_size=(1, 8), stride=1),
            torch.nn.BatchNorm2d(60),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(stride=1, kernel_size=(1, 2)))
        self.conv7 = torch.nn.Sequential(
            torch.nn.Conv2d(60, 60, kernel_size=(1,9), stride=1),
            torch.nn.BatchNorm2d(60),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=(1,2)))
        self.dense1 = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.Linear(60 * 1 * 219, 1024),
        )
        self.dense2 = torch.nn.Sequential(
            torch.nn.Linear(1024, num_class),
        )
        self.sf = torch.nn.Sequential(
            torch.nn.Softmax())

    # 前向传播
    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x6 = self.conv7(x6)
        x7 = x6.view(-1, 60 * 1 * 219)
        x8 = self.dense1(x7)
        x9 = self.dense2(x8)
        x10 = self.sf(x9)
        return x10

def train_model(model, criterion, optimizer, cls, num_epochs):
    best_acc = 0.0
    ctest = []
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:   # 每个epoch中都包含训练与验证两个阶段
            if phase == 'train':         # 训练阶段
                model.train()            # Set model to training mode
            else:                        # 测试阶段
                model.eval()             # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.   # 每个阶段都需要遍历所有的样本
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)   # 在维度1（行方向）查找最大值
                    labels = torch.max(labels, 1)[1]
                    loss = criterion(outputs, labels)  # 输出结果与label相比较
                    if phase == 'train':
                        loss.backward()     # 误差反向传播，计算每个w与b的更新值
                        optimizer.step()    # 将这些更新值施加到模型上
                # statistics
                running_loss += loss.item() * inputs.size(0)         # 计算当前epoch过程中，所有batch的损失和
                running_corrects += torch.sum(preds == labels.data)  # 判断正确的样本数

            epoch_loss = running_loss / cls[phase]               # 当前epoch的损失值是loss总和除以样本数
            epoch_acc = running_corrects.double() / cls[phase]   # 当前epoch的正确率

            if phase == 'val':
                ctest.append(epoch_acc)
            # deep copy the model
            if phase == 'val':                # 如果是val阶段，并且当前epoch的acc比best acc大
                best_acc = epoch_acc                                    # 就替换best acc为当前epoch的acc

    print('Best val Acc: {:4f}'.format(best_acc))                # 输出验证正确率 Best val Acc: 0.954248
    return best_acc

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 判断是否使用cpu

    model_ft = torch.load('./7_cnn_model.pt', map_location=torch.device('cpu'))
    model_ft = model_ft.to(device)  # 设置计算采用的设备，GPU还是CPU
    model_ft.dense2 = torch.nn.Linear(1024, 2, bias=True)
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9)
    flux = np.load('/$data_path$/')
    scls = np.load('/$label_path$/')

    data_loader_train, data_loader_test, clsTE, clsTR, batch_size = read_data(flux, scls)
    dataloaders = {'train': data_loader_train, 'val': data_loader_test}
    cls = {'train': clsTR, 'val': clsTE}
    acc = train_model(model_ft, criterion, optimizer_ft, None, cls, num_epochs=3000)    # 模型训练



