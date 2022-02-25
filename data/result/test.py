from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# acc = np.load('test_acc_node1.npy')
# print(acc)
#
# x = [i+1 for i in range(5)]
# y = list(map(lambda x: x ** 2, x))
# plt.plot(x, y, label='1111')
# plt.legend()
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
#
# x = np.linspace(0, 2, 100)
#
# plt.plot(x, x, label='linear')
# plt.plot(x, x ** 2, label='quadratic')
# plt.plot(x, x ** 3, label='cubic')
# plt.xlabel('x label')
# plt.xlabel('y label')
# plt.title('simple plot')
# plt.show()


# from multiprocessing import Pool, cpu_count

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.ndimage import gaussian_filter1d
#
# x=np.array([1,2,3,4,5,6,7])
# y=np.array([100,50,25,12.5,6.25,3.125,1.5625])
# y_smoothed = gaussian_filter1d(y, sigma=5)
#
# plt.plot(x, y_smoothed)
# plt.title("Spline Curve Using the Gaussian Smoothing")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()


# import numpy as np
# from scipy.interpolate import make_interp_spline
# import matplotlib.pyplot as plt
#
# x=np.array([1,2,3,4,5,6,7])
# y=np.array([100,50,25,12.5,6.25,3.125,1.5625])
#
# model=make_interp_spline(x, y)
#
# xs=np.linspace(1,7,500)
# ys=model(xs)
#
# plt.plot(xs, ys)
# plt.title("Smooth Spline Curve")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()


import pickle

# with open('FL_test_loss_all_epoch10.pkl', 'rb') as f:
#     loss = pickle.load(f)
# with open('FL_test_acc_all_epoch10.pkl', 'rb') as f:
#     acc = pickle.load(f)
#
# with open('./FL_test_loss_avg_epoch%d.pkl' % 10, 'wb') as f:
#     pickle.dump(loss['avg'], f)
# with open('./FL_test_acc_avg_epoch%d.pkl' % 10, 'wb') as f:
#     pickle.dump(acc['avg'], f)


# dic = {1:2, 3:4}
# print(dic.values())


# print(np.arange(10))
# a = [1,2,3,4,5]
# res = np.random.choice(a, size=len(a)+1, replace=True, p=[val/sum(a) for val in a])
# print(res[1])
# print(type(res))
# import datetime
# today = datetime.date.today()
# print('%s' % today)

dic1 = defaultdict(dict)
dic2 = defaultdict(list)
dic2[1].append(1)
print(dic2)
dic1[0] = dic2
print(dic1)

