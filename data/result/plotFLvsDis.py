import matplotlib.pyplot as plt
import pickle

from scipy.ndimage import gaussian_filter1d

# 测试精度
with open('./distributed10nodes/FL_test_acc_all_epoch100.pkl', 'rb') as f:
    y_dis = pickle.load(f)
with open('./FL10nodes/FL_test_acc_all_epoch100.pkl', 'rb') as f:
    y_fl = pickle.load(f)

# # 训练精度
# with open('./distributed10nodes/FL_train_acc_all_epoch100.pkl', 'rb') as f:
#     y_dis = pickle.load(f)
# with open('./FL10nodes/FL_train_acc_all_epoch100.pkl', 'rb') as f:
#     y_fl = pickle.load(f)

plt.figure() # 创建画布
for node in range(4):
    plt.subplot(2, 2, node+1) # 创建子图并指定位置
    x_data = [i + 1 for i in range(100)]
    plt.xticks([10*i for i in range(11)],) # 设置x轴显示的间隔、内容
    plt.ylim(0.745, 1.005)

    plt.xlabel('round', fontsize=10)
    plt.ylabel('acc', fontsize=10)
    plt.title('Node {} Test Accuracy'.format(node), fontsize=10)

    plt.plot(x_data, gaussian_filter1d(y_fl[node], sigma=1), label='FL', color='#1f77b4')
    plt.plot(x_data, gaussian_filter1d(y_dis[node], sigma=1), label='SL', color='#ff7f0e')

    plt.legend()

plt.show()