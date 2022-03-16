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
    EPOCHS = 2  # 总共训练批次
    BATCH_SIZE = 64
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRANSFORM = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    # 读取分割后的数据集
    data_path = r'./data/MNIST_data_nodes_100.pkl'
    with open(data_path, 'rb') as f:
        all_data = pickle.load(f)
    train_loaders = all_data['train_data']
    test_loader = all_data['test_data']

    # # 使用分割后前10各节点的训练数据集
    # train_dataset = train_loaders[0].dataset
    # for i in range(1, 10):
    #     train_dataset += train_loaders[i].dataset
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 读取全部训练集与分割后的1000张测试集
    train_dataset = datasets.MNIST('./data', train=True, transform=TRANSFORM, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # with open('./data/MNIST_onetenth_testloader.pkl', 'rb') as f:
    #     test_loader = pickle.load(f)


    # 初始化模型和优化器
    model = ConvNet().to(DEVICE)
    first_param = torch.load('./data/ModelParam.pth')
    model.load_state_dict(first_param)
    optimizer = optim.Adam(model.parameters())

    # 设置存储容器
    train_acc_central = []
    test_acc_central = []

    for epoch in range(1, EPOCHS + 1):
        _, train_acc = train(model, DEVICE, train_loader, optimizer, epoch, 'center')
        _, test_acc = test(model, DEVICE, test_loader)
        train_acc_central.append(train_acc)
        test_acc_central.append(test_acc)

    # 持久化存储
    save_path = './Central100nodes'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    with open('Central100nodes/train_acc_epoch100.pkl', 'wb') as f:
        pickle.dump(train_acc_central, f)
    with open('Central100nodes/test_acc_epoch100.pkl', 'wb') as f:
        pickle.dump(test_acc_central, f)

    # 压缩文件夹
    with ZipFile('Central100nodes.zip', 'w') as f:
        for file in os.listdir(save_path):
            f.write(os.path.join(save_path, file))

