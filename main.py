import datetime
import pickle
from collections import defaultdict
from copy import deepcopy
from multiprocessing import Pool

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import ConvNet, train, test, getAverageModel
from GMA import GeneticMergeAlg


def aggregate(params):
    """
    将各模型参数取平均，再赋值给每一个模型
    :param params: 含多个模型参数的列表
    :return:
    """
    print('\n>>> enter aggregation')
    with torch.no_grad():  # 就地操作需要no_grad包装
        new_param = []
        # 分层平均聚合
        for i in range(len(params[0])):
            tmp = [params[j][i] for j in range(len(params))]  # 取出各模型的第i层参数
            layer_param = sum(tmp) / len(params)
            new_param.append(layer_param)
        # new_param = [0.5 * params[0][i] + 0.5 * params[1][i] for i in range(len(params[0]))]
        # 分层替换原模型参数
        for param_index in range(len(params[0])):
            for i in range(len(params)):
                params[i][param_index].set_(deepcopy(new_param[param_index]))


def updataModels(models, new_model):
    # print('\n>>> Updata Models...')
    model_param = new_model.state_dict()
    for i in range(len(models)):
        models[i].load_state_dict(deepcopy(model_param))

def geneticFL(models, DEVICE, test_loader, pool, GENERATIONS, pm, pc, NP):
    # 遗传算法优化
    print('\n>>> GMA start ...')
    gma = GeneticMergeAlg(models, DEVICE, test_loader)
    gma_model = None # 遗传优化后要返回的模型
    generations_acc = []  # 存储每一代中最优个体的acc
    for i in range(GENERATIONS):
        print('\nGMA generation %d start \n' % i)
        gma.mutationInLayer(pm)
        gma.crossover(pc)
        fitness = gma.getFitness(pool)

        # 最后一代的最优个体作为当前epoch的中心节点gma聚合结果
        if i == GENERATIONS - 1:
            gma_acc = max(fitness)
            gma_model = deepcopy(gma.P[fitness.index(gma_acc)])
            generations_acc.append(gma_acc)
            print('\nGeneration {} best model\' acc: {}'.format(i, gma_acc))
            break

        best_fit = gma.tournamentSelection(3, NP)
        # best_fit = gma.rouletteSeletion(30)
        generations_acc.append(best_fit)
        print('\nGeneration {} best model\' acc: {}'.format(i, best_fit))
    return gma_model, generations_acc


if __name__ == '__main__':
    # 设置超参数
    CLIENT_NUM = 10
    EPOCHS = 10  # 总共训练批次
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GENERATIONS = 50

    # 读取分割后的数据集
    data_path = r'./data/MNIST_data_nodes_100.pkl'
    with open(data_path, 'rb') as f:
        all_data = pickle.load(f)
    train_loaders = all_data['train_data']
    test_loader = all_data['test_data']

    # 初始化模型和优化器
    models = [ConvNet().to(DEVICE) for _ in range(CLIENT_NUM)]
    optimizers = [optim.Adam(models[i].parameters()) for i in range(CLIENT_NUM)] # 针对model i 的优化器

    # 设置存储容器
    train_loss_all = defaultdict(list) # 所有参与方节点的训练损失
    train_acc_all = defaultdict(list) # 所有参与方节点的训练精度
    test_loss_all = defaultdict(list) # 所有参与方节点的测试损失
    test_acc_all = defaultdict(list) # 所有参与方节点的测试精度
    test_loss_center = defaultdict(list) # 中心节点测试损失，包括avg聚合、gma聚合
    test_acc_center = defaultdict(list) # 中心节点测试精度
    generations_test_acc = defaultdict(dict)

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

        # 联邦聚合，聚合后测试
        avg_model = getAverageModel(models)
        avg_loss, avg_acc = test(avg_model, DEVICE, test_loader, 'avg')
        # test_loss_center['avg'].append(avg_loss)
        test_acc_center['avg'].append(avg_acc)

        # 中心方测试精度优于参与方平均测试精度时进行遗传优化
        participants_now_acc = [test_acc_all[i][-1] for i in range(CLIENT_NUM)]
        if avg_acc >= sum(participants_now_acc) / CLIENT_NUM:
            gma_model, generations_acc = geneticFL(models, DEVICE, test_loader, po,
                                                   GENERATIONS=GENERATIONS, pm=0.5, pc=0.8, NP=30)
            test_acc_center['gma'].append(generations_acc[-1])
            generations_test_acc[epoch] = generations_acc
            best_model = gma_model
        else:
            best_model = avg_model
        updataModels(models, best_model) # 将最优的模型参数赋值为models

        cost_time = datetime.datetime.now() - start_time
        epoch_cost_time.append(cost_time)
        print('\nEpoch %d cost %f s\n' % (epoch, cost_time.seconds))

    po.close()
    po.join()

    test_loss_all.update(test_loss_center)
    test_acc_all.update(test_acc_center)

    today = datetime.date.today()
    with open('./data/result/%s/GMA_train_loss_all_epoch%d.pkl' % (today, EPOCHS), 'wb') as f:
        pickle.dump(train_loss_all, f)
    with open('./data/result/%s/GMA_train_acc_all_epoch%d.pkl' % (today, EPOCHS), 'wb') as f:
        pickle.dump(train_acc_all, f)
    with open('./data/result/%s/GMA_test_loss_all_epoch%d.pkl' % (today, EPOCHS), 'wb') as f:
        pickle.dump(test_loss_all, f)
    with open('./data/result/%s/GMA_test_acc_all_epoch%d.pkl' % (today, EPOCHS), 'wb') as f:
        pickle.dump(test_acc_all, f)
    with open('./data/result/%s/GMA_test_loss_center_epoch%d.pkl' % (today, EPOCHS), 'wb') as f:
        pickle.dump(test_loss_center, f)
    with open('./data/result/%s/GMA_test_acc_center_epoch%d.pkl' % (today, EPOCHS), 'wb') as f:
        pickle.dump(test_acc_center, f)
    with open('./data/result/%s/GMA_cost_time_epoch%d.pkl' % (today, EPOCHS), 'wb') as f:
        pickle.dump(epoch_cost_time, f)
    with open('./data/result/%s/GMA_generations_test_acc_epoch%d.pkl' % (today, EPOCHS), 'wb') as f:
        pickle.dump(generations_test_acc, f)