import matplotlib.pyplot as plt
import pickle


with open('GMA_test_acc_center_epoch10.pkl', 'rb') as f:
    y_data = pickle.load(f)
    y_gen = y_data['avg']
    # y_gen[-1] = 0.991
with open('FL_test_acc_all_epoch10.pkl', 'rb') as f:
    y_data = pickle.load(f)
    y_avg = y_data['avg']
with open('Central10nodes/test_acc_epoch10.pkl', 'rb') as f:
    y_cen = pickle.load(f)

x_data = [i + 1 for i in range(10)]
plt.xlabel('round', fontsize=12)
plt.xticks(x_data)
plt.ylabel('acc', fontsize=12)
plt.ylim(0.895, 1.005)
# plt.ylim(0.8, 1)
plt.title('Test Accuracy')

plt.plot(x_data, y_gen, label='FedGen')
plt.plot(x_data, y_avg, label='FedAvg')
plt.plot(x_data, y_cen, label='Central10nodes')

plt.legend()
plt.show()