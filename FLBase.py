import datetime
import pickle
from collections import defaultdict
from copy import deepcopy
from multiprocessing import Pool, cpu_count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# 模型定义
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1,28x28
        self.conv1 = nn.Conv2d(1, 10, 5)  # 24x24
        self.pool = nn.MaxPool2d(2, 2)  # 12x12
        self.conv2 = nn.Conv2d(10, 20, 3)  # 10x10
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x)  # 24
        out = F.relu(out)
        out = self.pool(out)  # 12
        out = self.conv2(out)  # 10
        out = F.relu(out)
        out = out.view(in_size, -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out


def train(model, device, train_loader, optimizer, epoch, node_num, train_loss_all=None):
    """
    训练函数定义
    :param model:
    :param device:
    :param train_loader:
    :param optimizer:
    :param epoch:
    :param node_num: 节点编号
    :param train_loss_all: 训练损失字典，全局变量
    :return: 训练精度
    """
    print('Node %d starts training...' % node_num)
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # _, id = torch.max(output.data, 1)
        # correct += torch.sum(id == target.data)
        pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
        correct += pred.eq(target.view_as(pred)).sum().item()

        if (batch_idx + 1) % 31 == 0:
            print('Node {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                node_num, epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx+1) / len(train_loader), loss.item()))
            # break
        if batch_idx == len(train_loader)-1:
            train_acc = correct / len(train_loader.dataset)
            print('Node {} Train Epoch: {}\tLoss: {:.6f}\tAcc: {:.6f}'.format(
                node_num, epoch, loss.item(), train_acc))
            # train_loss_all[node_num].append(loss.item())
            return loss.item()

def test(model, device, test_loader, node_num, test_loss_all=None, test_acc_all=None):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)
    print('\nNode {} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        node_num, test_loss, correct, len(test_loader.dataset),
        100. * test_acc))
    # test_loss_all[node_num].append(test_loss)
    # test_acc_all[node_num].append(test_acc)
    return test_loss, test_acc


def aggregate(params):
    """
    将各模型参数取平均，再赋值给每一个模型
    :param params: 含多个模型参数的列表
    :return:
    """
    print('\n>>> enter aggregation')
    with torch.no_grad():
        new_param = []
        # 分层平均聚合
        for i in range(len(params[0])):
            tmp = [params[j][i] for j in range(len(params))] # 取出各节点的第i层参数
            layer_param = sum(tmp) / len(params)
            new_param.append(layer_param)
        # new_param = [0.5 * params[0][i] + 0.5 * params[1][i] for i in range(len(params[0]))]
        # 分层替换原模型参数
        for param_index in range(len(params[0])):
            for i in range(len(params)):
                params[i][param_index].set_(deepcopy(new_param[param_index]))


if __name__ == '__main__':
    # 设置超参数
    CLIENT_NUM = 10
    EPOCHS = 50  # 总共训练批次
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取分割后的数据集
    data_path = r'./data/MNIST_data_nodes_%d.pickle' % CLIENT_NUM
    with open(data_path, 'rb') as f:
        client_data = pickle.load(f)
    train_loaders = client_data['train_data']
    test_loader = client_data['test_data']

    # 初始化模型和优化器
    models = [ConvNet().to(DEVICE) for _ in range(CLIENT_NUM)]
    optimizers = [optim.Adam(models[i].parameters()) for i in range(CLIENT_NUM)]

    # 保存数据
    train_loss_all = defaultdict(list) # 所有节点的训练损失
    test_loss_all = defaultdict(list) # 所有节点的测试损失
    test_acc_all = defaultdict(list) # 所有节点的测试精度
    test_loss_avg = [] # 中心节点测试损失
    test_acc_avg = [] # 中心节点测试精度

    """
    # 单进程
    for epoch in range(1, EPOCHS + 1):
        start_time = datetime.datetime.now()
        params = []
        for i in range(2):
            train(models[i], DEVICE, train_loaders[i], optimizers[i], epoch, i, train_loss_all)
            test(models[i], DEVICE, test_loader, i, test_loss_all, test_acc_all)
            params.append(list(models[i].parameters()))
        end_time = datetime.datetime.now()
        cost_time = end_time - start_time
        print('Epoch %d cost %f s' % (epoch, cost_time.seconds))
        aggregate(params)
        test(models[0], DEVICE, test_loader, 10, test_loss_all, test_acc_all) # 中心节点编号为10

    # 分别存储loss acc
    for key, val in train_loss_all.items():
        np.save('./data/result/train_loss_node%d.npy' % key, val)
    for key, val in test_loss_all.items():
        np.save('./data/result/test_loss_node%d.npy' % key, val)
    for key, val in test_acc_all.items():
        np.save('./data/result/test_acc_node%d.npy' % key, val)
    """

    # 多进程
    epoch_cost_time = []
    po = Pool()
    for epoch in range(1, EPOCHS + 1):
        start_time = datetime.datetime.now()
        train_res = []
        test_res = []

        # 多进程训练、测试
        for i in range(CLIENT_NUM):
            train_res.append(po.apply_async(train, args=(models[i], DEVICE, train_loaders[i], optimizers[i], epoch, i)))
        for i in range(CLIENT_NUM):
            test_res.append(po.apply_async(test, args=(models[i], DEVICE, test_loader, i)))

        # 保存loss acc
        for i in range(len(train_res)):
            train_loss_all[i].append(train_res[i].get())
        for i in range(len(test_res)):
            loss, acc = test_res[i].get()
            test_loss_all[i].append(loss)
            test_acc_all[i].append(acc)

        # 联邦聚合、测试
        params = [list(models[i].parameters()) for i in range(CLIENT_NUM)]
        aggregate(params)
        loss, acc = test(models[0], DEVICE, test_loader, 'avg') # 中心节点编号为 avg
        test_loss_avg.append(loss)
        test_acc_avg.append(acc)

        # 打印耗时
        cost_time = datetime.datetime.now() - start_time
        epoch_cost_time.append(cost_time)
        print('Epoch %d cost %f s\n' % (epoch, cost_time.seconds))

    po.close()
    po.join()

    # 将中心节点测试loss acc加入总字典
    test_loss_all['avg'] = test_loss_avg
    test_acc_all['avg'] = test_acc_avg

    with open('./data/result/FL_train_loss_all_epoch%d.pkl' % EPOCHS, 'wb') as f:
        pickle.dump(train_loss_all, f)
    with open('./data/result/FL_test_loss_all_epoch%d.pkl' % EPOCHS, 'wb') as f:
        pickle.dump(test_loss_all, f)
    with open('./data/result/FL_test_acc_all_epoch%d.pkl' % EPOCHS, 'wb') as f:
        pickle.dump(test_acc_all, f)
    with open('./data/result/FL_test_loss_avg_epoch%d.pkl' % EPOCHS, 'wb') as f:
        pickle.dump(test_loss_avg, f)
    with open('./data/result/FL_test_acc_avg_epoch%d.pkl' % EPOCHS, 'wb') as f:
        pickle.dump(test_acc_avg, f)
    with open('./data/result/FL_cost_time_epoch%d.pkl' % EPOCHS, 'wb') as f:
        pickle.dump(epoch_cost_time, f)
