import pickle

import torch
from torchvision import datasets, transforms

torch.manual_seed(1)  # cpu随机数种子
# 取MNIST原本测试集十分之一做测试集
TRANSFORM = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

test_dataset = datasets.MNIST('./', train=False, transform=TRANSFORM, download=True)
lengths = [9000, 1000] # 选取600张图片作为测试集, 600张作为验证集
data_split = torch.utils.data.random_split(dataset=test_dataset, lengths=lengths)
test_loader = torch.utils.data.DataLoader(data_split[0], batch_size=9000, shuffle=False)
val_loader = torch.utils.data.DataLoader(data_split[1], batch_size=1000, shuffle=False)
data = {
    'test_data':test_loader,
    'val_data':val_loader
}
with open('./MNIST_test_val_loader.pkl', 'wb') as f:
    pickle.dump(data, f)

# test_dataset = datasets.MNIST('../data', train=False, transform=TRANSFORM, download=True)
# onetenth_size = int(len(test_dataset) / 10)
# lengths = [onetenth_size for _ in range(10)]
# data_split = torch.utils.data.random_split(dataset=test_dataset, lengths=lengths)
# test_loader = torch.utils.data.DataLoader(data_split[0], batch_size=1000, shuffle=False)
#
# with open('./MNIST_onetenth_testloader.pkl', 'wb') as f:
#     pickle.dump(test_loader, f)