from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def train(model, device, train_loader, optimizer, epoch, node_num):
    """
    训练函数定义
    :param model:
    :param device:
    :param train_loader:
    :param optimizer:
    :param epoch:
    :param node_num: 节点编号
    :return: 训练精度
    """
    print('Node %d starts training...' % node_num)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 31 == 0:
            print('Node {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                node_num, epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx+1) / len(train_loader), loss.item()))
            # break
        if batch_idx == len(train_loader)-1:
            print('Node {} Train Epoch: {}\tFinal Loss: {:.6f}'.format(node_num, epoch, loss.item()))
            return loss.item()


def test(model, device, test_loader, node_num):
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
    # print('Node {} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    #     node_num, test_loss, correct, len(test_loader.dataset),
    #     100. * test_acc))
    return test_loss, test_acc


def getAverageModel(models):
    """
    联邦聚合生成平均模型
    :param models: 模型列表
    :return: 平均聚合后新的模型
    """
    params = [list(models[i].parameters()) for i in range(len(models))]
    avg_param = []
    # 分层平均聚合参数
    for i in range(len(params[0])):
        tmp = [params[j][i] for j in range(len(params))]  # 取出各模型的第i层参数
        layer_param = sum(tmp) / len(params)
        avg_param.append(layer_param)
    # 生成平均聚合模型
    avg_model = deepcopy(models[0])
    param = list(avg_model.parameters())
    with torch.no_grad():
        for param_idx in range(len(param)):
            param[param_idx].set_(avg_param[param_idx])
    return avg_model