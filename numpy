1.
import numpy as np
from torch import dtype
2.
a = np.array([4,5,6])
type(a) # numpy.ndarray
print(a.shape) # (3,)
print(a[0]) # 4
3.
b = np.array([[4,5,6],[1,2,3]])
print(b.shape) # (2, 3)
print(b[0,0]) # 4
print(b[0,1]) # 5
print(b[1,1]) # 2
4.
a = np.zeros((3,3),dtype=int)
b = np.ones((4,5),dtype=int)
c = np.identity(4)
d = np.random.randn(3,2)
5.6.
a = np.arange(1,13).reshape(3,4)
b = a[0:2, 1:3]
print(b[0, 0])
7.
c = a[1:3, :]
print(c[0][-1])
8.
a = np.array([[1,2],[3,4],[5,6]])
print(a[[0,1,2],[0,1,0]])
9.
a = np.arange(1,13).reshape(4,3)
b = np.array([0,2,0,1])
print(a[[np.arange(4),b]]) # [ 1  6  7 11]
10.
a[[np.arange(4),b]] += 10
var = a[[np.arange(4), b]]  # array([21, 26, 27, 31])
11.12.
x = np.array([1,2])
x =np.array([1.0,2.0])
print(x.dtype) # dtype('float64')
13.
x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)
print(x + y)
np.add(x,y)
14.
print(x-y)
np.subtract(x,y)
15.
print(x * y) # 两个矩阵对应位置元素相乘
np.multiply(x,y) # 两个矩阵对应位置元素相乘
np.dot(x,y) # 矩阵相乘
16.
print(x / y)
np.divide(x,y)
17.
np.sqrt(x)
18.
print(x.dot(y))
print(np.dot(x,y))
print(np.sum(x)) # 10
19.
print(np.sum(x,axis=0)) # [4. 6.] 两列之和
print(np.sum(x,axis=1)) # [3. 7.] 两行之和
20.
print(np.mean(x))
print(np.mean(x,axis=0))
print(np.mean(x,axis=1))
21.
print(x.T)
print(x.T)
22.
np.exp(x) # 求e的x次方的值
23.
print(np.argmax(x))
print(np.argmax(x,axis=0))
print(np.argmax(x,axis=1))
24.
import matplotlib.pyplot as plt
x = np.arange(0,100,0.1)
y = x * x
plt.figure(figsize=(6,6))  # 创建画布，并指定画布大小
plt.plot(x,y)   # 在画布上画图
plt.show()  # 展示画图结果
25.
x = np.arange(0,3*np.pi,0.1)
y1 = np.sin(x)
y2 = np.cos(x)
plt.figure(figsize=(10,6))
plt.plot(x,y1,color='Red')
plt.plot(x,y2,color='Blue')
plt.legend(['Sin','Cos'])  # 给两条线做标记
plt.show()
