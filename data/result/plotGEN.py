import matplotlib.pyplot as plt
import pickle

from scipy.ndimage import gaussian_filter1d

with open('./GEN10nodesNT/GMA_test_acc_all_epoch100.pkl', 'rb') as f:
# with open('./GEN10nodes/GMA_train_acc_all_epoch100.pkl', 'rb') as f:
    y_data = pickle.load(f)
    y_gen = y_data['gma']
    y_avg = y_data['avg']

with open('./GEN10nodesNT/GMA_val_acc_center_epoch100.pkl', 'rb') as f:
    y_val = pickle.load(f)

x_data = [i + 1 for i in range(100)]
plt.xticks([10*i for i in range(11)],) # 设置x轴显示的间隔、内容
plt.ylim(0.895, 1.005)

plt.xlabel('round', fontsize=12)
plt.ylabel('acc', fontsize=12)
plt.title('Test Accuracy')

plt.plot(x_data, gaussian_filter1d(y_gen, sigma=1), label='Gen', linewidth=2)
plt.plot(x_data, gaussian_filter1d(y_avg, sigma=1), label='Avg', linewidth=2)
# plt.plot(x_data, y_avg, label='FedGen', linewidth=2)
# for i in range(10):
#     plt.plot(x_data, gaussian_filter1d(y_data[i], sigma=1), label='Node {}'.format(i))

plt.legend()
plt.show()