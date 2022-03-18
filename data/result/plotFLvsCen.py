import matplotlib.pyplot as plt
import pickle

from scipy.ndimage import gaussian_filter1d
#
# # 验证精度
# with open('./FL10nodesNT3/FL_val_acc_avg_epoch100.pkl', 'rb') as f:
#     y_fl = pickle.load(f)
# # with open('Central10nodesNT5/val_acc_epoch100.pkl', 'rb') as f:
# #     y_cen = pickle.load(f)
# with open('Central100nodesNT/val_acc_epoch100.pkl', 'rb') as f:
#     y_cen = pickle.load(f)

# 测试精度
# with open('./GEN100nodes/GMA_test_acc_all_epoch100.pkl', 'rb') as f:
#     y_gen = pickle.load(f)
with open('./FL10nodesNT3/FL_test_acc_all_epoch100.pkl', 'rb') as f:
    y_fl = pickle.load(f)['avg']
    avg_acc = sum(y_fl)/100
    max_acc = max(y_fl)
    print(avg_acc, max_acc, y_fl.index(max_acc))
with open('Central100nodesNT/test_acc_epoch100.pkl', 'rb') as f:
# with open('Central10nodesNT5/test_acc_epoch100.pkl', 'rb') as f:
    y_cen = pickle.load(f)
    avg_acc = sum(y_cen)/100
    max_acc = max(y_cen)
    print(avg_acc, max_acc, y_cen.index(max_acc))

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

    # plt.plot(x_data, gaussian_filter1d(y_gen['gma'], sigma=1), label='FedGenFL')
    plt.plot(x_data, gaussian_filter1d(y_fl, sigma=1), label='FedAvgFL', color='#ff7f0e')
    plt.plot(x_data, gaussian_filter1d(y_cen, sigma=1), label='CL', color='#2ca02c')

    # plt.plot(x_data, y_gen['gma'], label='FedGen')
    # plt.plot(x_data, y_fl['avg'], label='FedAvg')
    # plt.plot(x_data, y_cen, label='Central')

    plt.legend(loc='lower right') # 标签位置

    plt.show()


plotCenter()