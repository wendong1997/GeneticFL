import pickle
import random
from copy import deepcopy
from multiprocessing import Pool

import torch
import numpy as np

from models import ConvNet, getAverageModel, test
from utils import changeOneBitInBinary

random.seed(2)


class GeneticMergeAlg(object):
    """遗传归并算法"""

    def __init__(self, models, DEVICE, test_loader):
        """
        :param models:
        """
        # 1.复制现有模型到新种群
        self.P = [deepcopy(models[i]) for i in range(len(models))]
        # 2.将平均模型纳入新种群
        avg_model = getAverageModel(models)
        self.P.append(avg_model)
        self.device = DEVICE
        self.test_loader = test_loader
        self.fitness = []

    def mutationInModel(self, pm):
        """
        变异：随机改变所选模型参数二进制表达式中的一位
        :param pm: 变异概率
        :return:
        """
        MP = [deepcopy(self.P[i]) for i in range(len(self.P))]
        layer_names = list(MP[0].state_dict().keys())
        for i in range(len(MP)):
            p = random.random()
            if p < pm:
                model = MP[i]

                # 随机挑选模型的某一层，在这一层的一维展开中随机选一个参数
                layer_name = layer_names[random.randint(0, len(layer_names) - 1)]
                layer_param = model.state_dict()[layer_name]
                layer_size = layer_param.size()
                layer_length = 1
                for size in layer_size:
                    layer_length *= size
                tmp = layer_param.reshape(layer_length)
                target_idx = random.randint(0, len(tmp) - 1)
                target = tmp[target_idx]

                # 随机修改该参数二进制表达式的一位
                target_change = changeOneBitInBinary(target)
                tmp[target_idx] = target_change
                # 一下两步可省略，也可修改模型参数
                layer_param = tmp.reshape(layer_size)
                model.state_dict()[layer_name] = deepcopy(layer_param)
        self.P.extend(MP)

    def mutationInLayer(self, pm):
        """
        变异：随机改变所选模型参数每一层中二进制表达式中的一位
        :param pm: 变异概率
        :return:
        """
        MP = [deepcopy(self.P[i]) for i in range(len(self.P))]
        layer_names = list(MP[0].state_dict().keys())
        for i in range(len(MP)):
            model = MP[i]
            for layer_name in layer_names:
                p = random.random()
                if p < pm:
                    layer_param = model.state_dict()[layer_name]
                    layer_size = layer_param.size()
                    layer_length = 1
                    for size in layer_size:
                        layer_length *= size
                    tmp = layer_param.reshape(layer_length)
                    target_idx = random.randint(0, len(tmp) - 1)
                    target = tmp[target_idx]

                    # 随机修改该参数二进制表达式的一位
                    target_change = changeOneBitInBinary(target)
                    tmp[target_idx] = target_change
                    # 一下两步可省略，也可修改模型参数
                    layer_param = tmp.reshape(layer_size)
                    model.state_dict()[layer_name] = deepcopy(layer_param)
        self.P.extend(MP)

    def crossover(self, pc):
        """
        交叉：交换P[i], P[i+1] 0<=i<len(P) 前L层的参数，将新的模型并入种群
        :param pc: 交叉概率
        :return:
        """
        CP = []
        for i in range(len(self.P)):
            # 以概率p2进行交叉
            p = random.random()
            if p < pc:
                model1 = deepcopy(self.P[i])
                model2 = deepcopy(self.P[(i + 1) % len(self.P)])
                model_key = list(model1.state_dict().keys())
                # 交换前一半的模型参数
                for j in range(len(model_key) // 2):
                    tmp = deepcopy(model1.state_dict()[model_key[j]])
                    model1.state_dict()[model_key[j]].copy_(model2.state_dict()[model_key[j]])
                    model2.state_dict()[model_key[j]].copy_(tmp)
                CP.append(model1)
                CP.append(model2)
        self.P.extend(CP)

    def getFitness(self, pool):
        callback = []
        fitness = [] # [i, P[i]的test_acc]
        for i in range(len(self.P)):
            callback.append(pool.apply_async(test, args=(self.P[i], self.device, self.test_loader, i)))
        for j in range(len(callback)):
            _, tmp_acc = callback[j].get()
            fitness.append(tmp_acc)
        self.fitness = fitness
        return fitness

    def select(self, po, device, test_loader):
        callback = []
        res = []
        best_idx = 0
        min_loss, max_acc = 1, 0
        for i in range(len(self.P)):
            callback.append(po.apply_async(test, args=(self.P[i], device, test_loader, i)))
        for j in range(len(callback)):
            tmp_loss, tmp_acc = callback[j].get()
            res.append([j, tmp_acc])
            if tmp_acc > max_acc:
                max_acc = tmp_acc
                min_loss = tmp_loss
                best_idx = j
        best_model = deepcopy(self.P[best_idx])
        test_acc = [item[1] for item in res]

        # 轮盘赌选择
        new_idx = np.random.choice(range(len(self.P)), size=30, replace=True,
                         p=[acc/sum(test_acc) for acc in test_acc])
        new_p = [self.P[idx] for idx in new_idx]
        new_p.append(self.P[best_idx])
        self.P = new_p
        return best_model, min_loss, max_acc

    def rouletteSeletion(self, population_size):
        fitness = deepcopy(self.fitness)

        # 轮盘赌选择
        new_idx = np.random.choice(range(len(self.P)), size=population_size-1, replace=True,
                         p=[fit/sum(fitness) for fit in fitness])
        new_p = [self.P[idx] for idx in new_idx]
        best_idx = fitness.index(max(fitness))
        best_individual = self.P[best_idx]
        new_p.append(best_individual) # self.P最后一个即为最优个体
        self.P = new_p
        return fitness[best_idx]

    def tournamentSelection(self, tournament_size, population_size):
        fitness = deepcopy(self.fitness)
        best_fit = 0

        # 锦标赛选择
        population = list(range(len(self.P))) # 种群个体下标
        complete = lambda competitors: max(competitors, key=fitness.__getitem__)
        new_p = []
        while population and len(new_p) < population_size:
            competitors = random.sample(population, min(tournament_size, len(population))) # 随机选取一批竞争对手
            winner = complete(competitors)
            best_fit = max(best_fit, fitness[winner])
            new_p.append(self.P[winner]) # 胜者加入下一轮
            population.remove(winner) # 将胜者从当前代中剔除: remove删除的是值，与下标顺序无关
        self.P = new_p
        return best_fit


if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [ConvNet().to(DEVICE) for _ in range(10)]
    with open('./data/MNIST_onetenth_testloader.pkl', 'rb') as f:
        test_loader = pickle.load(f)
    po = Pool()
    gma = GeneticMergeAlg(models, DEVICE, test_loader, po)
    gma.mutationInLayer(1)
    gma.crossover(0.8)
    best_acc = gma.tournamentSelection(3, 20)

    po.close()
    po.join()
