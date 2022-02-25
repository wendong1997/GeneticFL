import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d
import numpy as np
import pickle


def readSeparately():
    NODE_NUM = 2
    acc_all = [np.load('test_acc_node%d.npy' % i) for i in range(NODE_NUM)]
    acc_all.append(np.load('test_acc_node10.npy'))
    x_data = [i + 1 for i in range(len(acc_all[0]))]
    # print(data)
    # print(x_data)
    plt.xlabel("x_data", fontdict={'size': 12})  # x轴命名
    plt.ylabel("acc", size=12)  # y轴命名
    plt.xticks(ticks=x_data) # 设置x轴范围、间隔
    plt.title('Test Accuracy')
    for i in range(len(acc_all)):
        plt.plot(x_data, acc_all[i], label='Node %d' % i, linewidth=1)  # 横坐标列表，纵坐标列表，线宽
    plt.legend() # 显示图例
    plt.show()

def plotResult(path, x_label, y_label, title):
    with open(path, 'rb') as f:
        y_data = pickle.load(f)

    x_data = [i+1 for i in range(len(list(y_data.values())[0]))]
    plt.xlabel(x_label, fontsize=12)
    plt.xticks(x_data)
    plt.ylabel(y_label, fontsize=12)
    # plt.ylim(-0.05, 1.05)
    plt.ylim(0.8, 1)
    plt.title(title)
    print(y_data)
    for k, v in y_data.items():
        plt.plot(x_data, v, label='Node {}'.format(k))
        # plt.plot(x_data, gaussian_filter1d(y_data[i], sigma=1), label='Node %d' % i) # 曲线平滑
    plt.legend()
    plt.show()

def plotTrainLoss(path):
    plotResult(path,  'round', 'loss', 'Train Loss')

def plotTestLoss(path):
    plotResult(path,  'round', 'loss', 'Test Loss')

def plotTestAcc(path):
    plotResult(path,  'round', 'acc', 'Test Accuracy')


if __name__ == '__main__':
    # plotResult('train_loss_all.pkl', 'round', 'loss', 'Train Loss')
    # plotResult('test_loss_all.pkl', 'round', 'loss', 'Test Loss')
    # plotResult('test_acc_all.pkl', 'round', 'acc', 'Test Accuracy')

    # plotResult('FL_test_acc_all_epoch10.pkl', 'round', 'acc', 'Test Accuracy')
    # plotResult('FL_test_loss_all_epoch10.pkl', 'round', 'loss', 'Test Loss')
    # plotResult('FL_train_loss_all_epoch10.pkl', 'round', 'loss', 'Train Loss')

    # plotTestAcc('GMA_test_acc_all_epoch10.pkl')
    plotTestAcc('GMA_test_acc_center_epoch10.pkl')
