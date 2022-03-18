import pickle

import torch
from torchvision import datasets, transforms


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
        CLIENT_NUM = 10
        self.root = './'
        self.epoch_exchange = 10

# 设置超参数
torch.manual_seed(1)  # cpu随机数种子
CLIENT_NUM = 100
BATCH_SIZE = 64  # 批次大小
TEST_BATCH_SIZE = 1200
VAL_BATCH_SIZE = 600
TRANSFORM = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

# 分割训练集
train_dataset = datasets.MNIST('./', train=True, transform=TRANSFORM, download=True)
subset_size = int(len(train_dataset) / CLIENT_NUM)
lengths = [subset_size for _ in range(CLIENT_NUM)]
data_split = torch.utils.data.random_split(dataset=train_dataset, lengths=lengths)
train_loader_list = []
for i in range(CLIENT_NUM):
    train_loader = torch.utils.data.DataLoader(data_split[i], batch_size=BATCH_SIZE, shuffle=True)
    train_loader_list.append(train_loader)

# 分割测试集
test_dataset = datasets.MNIST('./', train=False, transform=TRANSFORM, download=True)
lengths = [1200, 600, 8200] # 选取600张图片作为测试集, 600张作为验证集
data_split = torch.utils.data.random_split(dataset=test_dataset, lengths=lengths)
test_loader = torch.utils.data.DataLoader(data_split[0], batch_size=TEST_BATCH_SIZE, shuffle=False)
val_loader = torch.utils.data.DataLoader(data_split[1], batch_size=VAL_BATCH_SIZE, shuffle=False)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

# 保存
save_data = {
    "train_data": train_loader_list,
    "test_data": test_loader,
    "val_data": val_loader
}
filedirname = './MNIST_data_nodes_%d.pkl' % CLIENT_NUM
with open(filedirname, 'wb') as f:
    pickle.dump(save_data, f)
