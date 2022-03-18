import matplotlib.pyplot as plt
import pickle

from scipy.ndimage import gaussian_filter1d

# 测试精度
with open('./GEN10nodes/GMA_test_acc_all_epoch100.pkl', 'rb') as f:
    y_gen = pickle.load(f)
with open('./FL10nodesNT2/FL_test_acc_all_epoch100.pkl', 'rb') as f:
    y_fl = pickle.load(f)
with open('Central10nodesNT5/test_acc_epoch100.pkl', 'rb') as f:
    y_cen = pickle.load(f)
    # y_cen = [item+0.001 for item in y_cen]
# with open('./GEN10nodesNew/GMA_val_acc_center_epoch100.pkl', 'rb') as f:
#     y_val = pickle.load(f)

# # 训练精度
# with open('./distributed10nodes/FL_train_acc_all_epoch100.pkl', 'rb') as f:
#     y_dis = pickle.load(f)
# with open('./FL10nodes/FL_train_acc_all_epoch100.pkl', 'rb') as f:
#     y_fl = pickle.load(f)



def plotCenter():
    x_data = [i + 1 for i in range(100)]
    plt.xticks([10*i for i in range(11)],) # 设置x轴显示的间隔、内容
    plt.ylim(0.895, 1.005)

    plt.xlabel('round', fontsize=10)
    plt.ylabel('acc', fontsize=10)
    plt.title('Node Center Test Accuracy', fontsize=10)

    plt.plot(x_data, gaussian_filter1d(y_gen['gma'], sigma=1), label='FedGenFL')
    # plt.plot(x_data, gaussian_filter1d(y_val, sigma=1), label='FedAvlFL')
    plt.plot(x_data, gaussian_filter1d(y_fl['avg'], sigma=1), label='FedAvgFL')
    plt.plot(x_data, gaussian_filter1d(y_cen, sigma=1), label='CL')

    # plt.plot(x_data, y_gen['gma'], label='FedGenFL')
    # plt.plot(x_data, y_fl['avg'], label='FedAvgFL')
    # plt.plot(x_data, y_cen, label='CL')

    plt.legend(loc='lower right') # 标签位置

    plt.show()


plotCenter()