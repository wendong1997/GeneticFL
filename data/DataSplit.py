import pickle

import torch
from torchvision import datasets, transforms


# 生成n个节点的数据，讲训练集的数据评分到n个节点上

class Arguments():
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 1000
        self.epochs = 40
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 30
        self.save_model = False
        self.clients = 10
        self.root = './'
        self.epoch_exchange = 10


# 设置超参数
args = Arguments()
torch.manual_seed(args.seed)  # cpu随机数种子
TRANSFORM = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])
BATCH_SIZE = 512  # 批次大小


# 分割数据集
train_dataset = datasets.MNIST('./data', train=True, transform=TRANSFORM, download=True)
subset_size = int(len(train_dataset) / args.clients)
lengths = [subset_size for _ in range(args.clients)]
# length[-1] = int(len(train_dataset) - (args.clients - 1) * subset_size)
data_split = torch.utils.data.random_split(dataset=train_dataset, lengths=lengths)

train_loader_list = []
for i in range(args.clients):
    train_loader = torch.utils.data.DataLoader(data_split[i], batch_size=args.batch_size, shuffle=True)
    train_loader_list.append(train_loader)

test_dataset = datasets.MNIST('./data', train=False, transform=TRANSFORM, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

savetarget = {
    "train_data": train_loader_list,
    "test_data": test_loader
}

filedirname = r'./data/MNIST_data_nodes_%d.pickle' % args.clients
with open(filedirname, 'wb') as f:
    pickle.dump(savetarget, f)
