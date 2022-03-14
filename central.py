import os
import datetime
import pickle
from collections import defaultdict
from copy import deepcopy
from multiprocessing import Pool
from zipfile import ZipFile

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from model import ConvNet, train, test, getAverageModel
from GMA import GeneticMergeAlg


if __name__ == '__main__':
    # 设置超参数
    EPOCHS = 10  # 总共训练批次
    BATCH_SIZE = 64
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRANSFORM = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    # 读取全部训练集与分割后的1000张测试集
    train_dataset = datasets.MNIST('./data', train=True, transform=TRANSFORM, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    with open('./data/MNIST_onetenth_testloader.pkl', 'rb') as f:
        test_loader = pickle.load(f)

    # 初始化模型和优化器
    model = ConvNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters())

    # 设置存储容器
    train_acc_central = []
    test_acc_central = []

    for epoch in range(1, EPOCHS + 1):
        _, train_acc = train(model, DEVICE, train_loader, optimizer, epoch, 'center')
        _, test_acc = test(model, DEVICE, test_loader)
        train_acc_central.append(train_acc)
        test_acc_central.append(test_acc)

    with open('./data/result/Central/train_acc_epoch10.pkl', 'wb') as f:
        pickle.dump(train_acc_central, f)
    with open('./data/result/Central/test_acc_epoch10.pkl', 'wb') as f:
        pickle.dump(test_acc_central, f)


