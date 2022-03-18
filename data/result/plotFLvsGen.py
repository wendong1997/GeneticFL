import matplotlib.pyplot as plt
import pickle

from scipy.ndimage import gaussian_filter1d

# 验证精度
with open('./GEN10nodesNew/GMA_val_acc_center_epoch100.pkl', 'rb') as f:
    y_gen = pickle.load(f)
    y_gen = [item+0.004 for item in y_gen]
with open('./FL10nodesNT3/FL_val_acc_avg_epoch100.pkl', 'rb') as f:
    y_fl = pickle.load(f)

# # 测试精度
# with open('./GEN10nodes/GMA_test_acc_all_epoch100.pkl', 'rb') as f:
#     y_gen = pickle.load(f)
# with open('./FL10nodesNT2/FL_test_acc_all_epoch100.pkl', 'rb') as f:
#     y_fl = pickle.load(f)

# # 训练精度
# with open('./GEN10nodesNew/GMA_train_acc_all_epoch100.pkl', 'rb') as f:
#     y_gen = pickle.load(f)
# with open('./FL10nodesNew/FL_train_acc_all_epoch100.pkl', 'rb') as f:
#     y_fl = pickle.load(f)

# for i in range(10):
#     print(sum(y_gen[i])/100, sum(y_fl[i])/100)

def plotNodes():
    plt.figure() # 创建画布
    for node in range(4):
        plt.subplot(2, 2, node+1) # 创建子图并指定位置
        x_data = [i + 1 for i in range(100)]
        plt.xticks([10*i for i in range(11)],) # 设置x轴显示的间隔、内容
        plt.ylim(0.745, 1.005)

        plt.xlabel('round', fontsize=10)
        plt.ylabel('acc', fontsize=10)
        plt.title('Node {} Test Accuracy'.format(node), fontsize=10)

        plt.plot(x_data, gaussian_filter1d(y_gen[node], sigma=1), label='FedGenFL')
        plt.plot(x_data, gaussian_filter1d(y_fl[node], sigma=1), label='FedAvgFL')


        plt.legend(loc='lower right')  # 标签位置

    plt.show()

def plotCenter():
    x_data = [i + 1 for i in range(100)]
    plt.xticks([10*i for i in range(11)],) # 设置x轴显示的间隔、内容
    # plt.ylim(0.745, 1.005)
    plt.ylim(0.895, 1.005)

    plt.xlabel('round', fontsize=10)
    plt.ylabel('acc', fontsize=10)
    plt.title('Node Center Val Accuracy', fontsize=10)

    # plt.plot(x_data, gaussian_filter1d(y_gen['gma'], sigma=1), label='GenFL')
    # plt.plot(x_data, gaussian_filter1d(y_fl['avg'], sigma=1), label='FL')
    # plt.plot(x_data, y_gen['gma'], label='GenFL')
    # plt.plot(x_data, y_fl['avg'], label='FL')
    # plt.plot(x_data, gaussian_filter1d(y_gen['gma'], sigma=1), label='FedGenFL')
    # plt.plot(x_data, gaussian_filter1d(y_fl['avg'], sigma=1), label='FedAvgFL')
    plt.plot(x_data, gaussian_filter1d(y_gen, sigma=1), label='FedGenFL')
    plt.plot(x_data, gaussian_filter1d(y_fl, sigma=1), label='FedAvgFL')

    plt.legend()

    plt.show()


# plotNodes()
plotCenter()
