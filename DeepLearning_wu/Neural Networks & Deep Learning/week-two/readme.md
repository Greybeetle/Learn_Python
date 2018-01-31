
# python and vectorization
## 向量化和for循环的区别


```python
import numpy as np
a=np.array([1,2,3,4])
print(a)
```

    [1 2 3 4]
    


```python
import time
a=np.random.rand(1000000)
b=np.random.rand(1000000)
tic=time.time()
c=np.dot(a,b)
toc=time.time()
print(c)
print("vectorized version:"+str(1000*(toc-tic))+"ms")
c=0
tic=time.time()
for i in range(1000000):
    c+=a[i]*b[i]
toc=time.time()
print(c)
print("For loop version:"+str(1000*(toc-tic))+"ms")
```

    249961.213603
    vectorized version:2.006053924560547ms
    249961.213603
    For loop version:624.1602897644043ms
    

***可以看出，向量化之后的运行速度是for循环的接近300倍***

## python中的广播

以例子的形式说明：  
![](https://ws1.sinaimg.cn/large/c2894cd5gy1fmho5v1m3bj21f90mrtor.jpg)


```python
import numpy as np
A=np.array([[56.0, 0.0, 4.4, 68.0],
          [1.2, 104.0, 52.0, 8.0],
          [1.8, 135.0, 99.0, 0.9]])
print(A)
```

    [[  56.     0.     4.4   68. ]
     [   1.2  104.    52.     8. ]
     [   1.8  135.    99.     0.9]]
    


```python
cal = A.sum(axis=0)
print(cal)
```

    [  59.   239.   155.4   76.9]
    


```python
percentage=100*A/cal.reshape(1,4)
print(percentage)
```

    [[ 94.91525424   0.           2.83140283  88.42652796]
     [  2.03389831  43.51464435  33.46203346  10.40312094]
     [  3.05084746  56.48535565  63.70656371   1.17035111]]
    

## 作业

![](https://ws1.sinaimg.cn/large/c2894cd5gy1fmjv4mhvovj21120knq5h.jpg)  
![](https://ws1.sinaimg.cn/large/c2894cd5gy1fmjv5t6xsoj20s10mkq4v.jpg)  
![](https://ws1.sinaimg.cn/large/c2894cd5gy1fmjv6hu8lvj20sc0lydhh.jpg)  
![](https://ws1.sinaimg.cn/large/c2894cd5gy1fmjv7cokipj20rs0niwgj.jpg)  
![](https://ws1.sinaimg.cn/large/c2894cd5gy1fmjv8ranz6j20to0n0mzj.jpg)


```python
import numpy as np
a = np.random.randn(2, 3) # a.shape = (2, 3)
b = np.random.randn(2, 1) # b.shape = (2, 1)
c = a + b
print(c.shape)
```

    (2, 3)
    


```python
a = np.random.randn(4, 3) # a.shape = (4, 3)
b = np.random.randn(3, 2) # b.shape = (3, 2)
c = a*b
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-4-4bd2cd4ebeba> in <module>()
          1 a = np.random.randn(4, 3) # a.shape = (4, 3)
          2 b = np.random.randn(3, 2) # b.shape = (3, 2)
    ----> 3 c = a*b
    

    ValueError: operands could not be broadcast together with shapes (4,3) (3,2) 



```python
a = np.random.randn(12288, 150) # a.shape = (12288, 150)
b = np.random.randn(150, 45) # b.shape = (150, 45)
c = np.dot(a,b)
print(c.shape)
```

    (12288, 45)
    


```python
a = np.random.randn(3, 3)
b = np.random.randn(3, 1)
c = a*b
print(c.shape)
```

    (3, 3)
    


```python
a=np.zeros((4,1),dtype=np.float64)
assert(a.shape==(4,1))
```


```python
b=np.zeros((1,),dtype=float)
print(b.dtype)
assert(isinstance(b, float) or isinstance(b, int))
```

    float64
    


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-31-74b7a711bfd7> in <module>()
          1 b=np.zeros((1,),dtype=float)
          2 print(b.dtype)
    ----> 3 assert(isinstance(b, float) or isinstance(b, int))
    

    AssertionError: 

