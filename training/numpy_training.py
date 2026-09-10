import numpy as np

x = np.array([1, 2, 1, 3])
w= np.where(x == 1)
# print(w)
# print(type(w))
# max_value = np.max(x)
# print(max_value)

def arr(a):
    print(a)
    print(a.shape)
    return a

x = np.array([x])
arr(x)