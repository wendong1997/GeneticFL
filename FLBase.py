import datetime
import os
import pickle
from collections import defaultdict
from copy import deepcopy
from multiprocessing import Pool, cpu_count
from zipfile import ZipFile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from model import ConvNet, train, test


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
    data_path = r'./data/MNIST_data_nodes_100.pkl'
    with open(data_path, 'rb') as f:
        all_data = pickle.load(f)
    train_loaders = all_data['train_data']
    test_loader = all_data['test_data']

    # 初始化模型和优化器
    models = [ConvNet().to(DEVICE) for _ in range(CLIENT_NUM)]
    optimizers = [optim.Adam(models[i].parameters()) for i in range(CLIENT_NUM)]

    # 设置存储容器
    train_loss_all = defaultdict(list) # 所有参与方节点的训练损失
    train_acc_all = defaultdict(list) # 所有参与方节点的训练精度
    test_loss_all = defaultdict(list) # 所有参与方节点的测试损失
    test_acc_all = defaultdict(list) # 所有参与方节点的测试精度
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

        # 多进程模拟参与方节点训练、测试
        for i in range(CLIENT_NUM):
            train_res.append(po.apply_async(train, args=(models[i], DEVICE, train_loaders[i], optimizers[i], epoch, i)))
        for i in range(CLIENT_NUM):
            test_res.append(po.apply_async(test, args=(models[i], DEVICE, test_loader, i)))

        # 保存参与节点训练集和测试集的loss acc
        for i in range(len(train_res)):
            train_loss, train_acc = train_res[i].get()
            train_loss_all[i].append(train_loss)
            train_acc_all[i].append(train_acc)
        for i in range(len(test_res)):
            test_loss, test_acc = test_res[i].get()
            test_loss_all[i].append(test_loss)
            test_acc_all[i].append(test_acc)

        # 联邦聚合、测试
        params = [list(models[i].parameters()) for i in range(CLIENT_NUM)]
        aggregate(params)
        loss, acc = test(models[0], DEVICE, test_loader, 'avg') # 中心节点编号为 avg
        print('Avg Model\'s Loss: {:.6f}\tAcc: {:.6f}'.format(loss, acc))
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

    # 持久化存储
    save_path = './FL'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    with open('./FL/FL_train_loss_all_epoch%d.pkl' % EPOCHS, 'wb') as f:
        pickle.dump(train_loss_all, f)
    with open('./FL/FL_train_acc_all_epoch%d.pkl' % EPOCHS, 'wb') as f:
        pickle.dump(train_acc_all, f)
    with open('./FL/FL_test_loss_all_epoch%d.pkl' % EPOCHS, 'wb') as f:
        pickle.dump(test_loss_all, f)
    with open('./FL/FL_test_acc_all_epoch%d.pkl' % EPOCHS, 'wb') as f:
        pickle.dump(test_acc_all, f)
    with open('./FL/FL_test_loss_avg_epoch%d.pkl' % EPOCHS, 'wb') as f:
        pickle.dump(test_loss_avg, f)
    with open('./FL/FL_test_acc_avg_epoch%d.pkl' % EPOCHS, 'wb') as f:
        pickle.dump(test_acc_avg, f)
    with open('./FL/FL_cost_time_epoch%d.pkl' % EPOCHS, 'wb') as f:
        pickle.dump(epoch_cost_time, f)

    # 压缩文件夹
    with ZipFile('FL.zip', 'w') as f:
        for file in os.listdir(save_path):
            f.write(os.path.join(save_path, file))