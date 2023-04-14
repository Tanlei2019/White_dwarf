from glob import glob
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os
import csv
import torch
import sys
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import torch.utils.data as Data
import warnings
from torch.autograd import Variable
from scipy.ndimage import median_filter
warnings.filterwarnings("ignore")
import os
import concurrent.futures

def get_variable(x):
    x = Variable(x)
    return x

def read_lamost(paths, flux, scls):
    paths = glob(paths)
    wav = 3700
    for idx, file in enumerate(paths):
        with fits.open(file) as hdulist:
            snrg = hdulist[0].header['SNRG']
            f = hdulist[1].data[0][0]
            f = f[:wav]
            f = (f - np.min(f)) / (np.max(f) - np.min(f))
            f = median_filter(f, size=9, mode='reflect')
            f = np.array([np.array([f])])
            s = hdulist[0].header['class'][0]
        flux.append(f)
        scls.append(s)
    return flux, scls, snrg

def run_CNN_module(cnn_module, test):
    # 对模型进行训练和参数优化
    cnn_model = cnn_module.eval()
    pred = np.array([])
    a0 = np.array([])
    for data in test:
        X_test, y_test = data
        X_test, y_test = get_variable(X_test), get_variable(y_test)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        outputs = cnn_model(X_test)
        a, pre = torch.max(outputs.data, 1)
        a = np.array(a)
        a = np.array(a)
        pre = np.array(pre)
        pred = np.append(pred, pre)
        a0 = np.append(a0, a)
    return a0, pred
def read_data(paths):
    flux = []
    scls = []
    flux, scls, snrg = read_lamost(paths, flux, scls)
    flux = np.array(flux)
    cls = onehot(scls)
    Xtest1 = torch.from_numpy(flux)
    ytest1 = torch.from_numpy(cls)
    torch_dataset_test = Data.TensorDataset(Xtest1, ytest1)
    data_loader_test = torch.utils.data.DataLoader(dataset=torch_dataset_test, batch_size=1, shuffle=False)
    return data_loader_test, scls, snrg

def onehot(classes):
    ''' Encodes a list of descriptive labels as one hot vectors '''
    label_encoder = LabelEncoder()
    int_encoded = label_encoder.fit_transform(classes)
    labels = label_encoder.inverse_transform(np.arange(np.amax(int_encoded) + 1))
    onehot_encoder = OneHotEncoder(sparse=False)
    int_encoded = int_encoded.reshape(len(int_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(int_encoded)
    return onehot_encoded

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


def move(path, cnn_model):
    test, scls, snrg = read_data(path)
    hdulist = fits.open(path, 'update')
    a, pred = run_CNN_module(cnn_model,  test)
    pred = int(pred[0])
    if pred == a:# 7分类时a=6，2分类时a=1
        filename = os.path.basename(path)
        filepath = '/$candidates_path$/'
        filename = os.path.join(filepath, filename)
        hdulist.writeto(filename, overwrite=True, output_verify='ignore')
        hdulist.close(output_verify='ignore')

cnn_model = torch.load('./x_cnn_model.pt', # x在7分类时选7，在二分类时2
                       map_location=torch.device('cpu'))

t = '/$data_path$/*.fits'
paths = glob(t)
for path in paths:
    move(path, cnn_model)

