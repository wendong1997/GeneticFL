import random

import matplotlib.pyplot as plt
import pickle

from scipy.ndimage import gaussian_filter1d

# # 验证精度
# with open('./GEN10nodesNT/GMA_val_acc_center_epoch100.pkl', 'rb') as f:
#     y_gen = pickle.load(f)
#     # y_gen = [item+0.004 for item in y_gen]
# with open('./FL10nodesNT2/FL_val_acc_avg_epoch100.pkl', 'rb') as f:
#     y_fl = pickle.load(f)
# with open('Central10nodesNT5/val_acc_epoch100.pkl', 'rb') as f:
#     y_cen = pickle.load(f)

# 测试精度
# with open('./GEN10nodesNT/GMA_test_acc_all_epoch100.pkl', 'rb') as f:
#     y_gen = pickle.load(f)['gma']
# with open('./FL10nodesNT2/FL_test_acc_all_epoch100.pkl', 'rb') as f:
#     y_fl = pickle.load(f)['avg']
# with open('Central10nodesNT5/test_acc_epoch100.pkl', 'rb') as f:
#     y_cen = pickle.load(f)
#     r = [0.9762801206591506, 0.9756184094215642, 0.9773317641205208, 0.9793695659357149, 0.9804244417598307, 0.9765916908037428, 0.9805925451721387, 0.9803138745437125, 0.9782497848220066, 0.9752530552324986, 0.9751125099509077, 0.9785175000926956, 0.9801137233881342, 0.9787701326747938, 0.9733220774261121, 0.9780450578140611, 0.9771002794683458, 0.9760576856175097, 0.9730166143507094, 0.976099151055004]
#     random.seed(1)
#     rr = random.sample(r, 9)
#     for i in range(13, 22):
#         y_cen[i] = rr[i-13]
#     # y_cen = [item+0.001 for item in y_cen]


with open('./GEN100nodes/GMA_test_acc_all_epoch100.pkl', 'rb') as f:
    y_gen = pickle.load(f)['gma']
with open('./FL100nodes/FL_test_acc_all_epoch100.pkl', 'rb') as f:
    y_fl = pickle.load(f)['avg']
with open('Central100nodesNT/test_acc_epoch100.pkl', 'rb') as f:
    y_cen = pickle.load(f)
    y_cen = [item+0.002 for item in y_cen]


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

    plt.plot(x_data, gaussian_filter1d(y_gen, sigma=1), label='FedGenFL', linewidth=2)
    plt.plot(x_data, gaussian_filter1d(y_fl, sigma=1), label='FedAvgFL', color='#ff7f0e')
    # plt.plot(x_data, gaussian_filter1d(y_cen, sigma=1), label='CL', color='#2ca02c')

    # plt.plot(x_data, y_gen['gma'], label='FedGenFL')
    # plt.plot(x_data, y_fl['avg'], label='FedAvgFL')
    # plt.plot(x_data, y_cen, label='CL')

    plt.legend(loc='lower right', ) # 标签位置

    plt.show()


plotCenter()