import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def ya1():
    A = np.array([[1.0,2.0],[1.0,0.0],[2.0,3.0]])
    A_t = A.transpose()
    print("A:")
    print(A)
    print("A 的转置:")
    print(A_t)

    a = np.array([[1.0,2.0],[3.0,4.0]])
    b = np.array([[6.0,7.0],[8.0,9.0]])
    print("矩阵相加：", a + b)

    m1 = np.array([[1.0, 3.0], [1.0, 0.0]])
    m2 = np.array([[1.0, 2.0], [5.0, 0.0]])
    print("按矩阵乘法规则：", np.dot(m1, m2))
    print("按逐元素相乘：", np.multiply(m1, m2))
    print("按逐元素相乘：", m1 * m2)

    v1 = np.array([1.0, 2.0])
    v2 = np.array([4.0, 5.0])
    print("向量内积：", np.dot(v1, v2))

    print("单位矩阵：",np.identity(3))

    A = [[1.0, 2.0], [3.0, 4.0]]
    A_inv = np.linalg.inv(A)
    print("A 的逆矩阵", A_inv)

    a = np.array([1.0,3.0])
    print("向量 2 范数", np.linalg.norm(a,ord=2))
    print("向量 1 范数", np.linalg.norm(a,ord=1))
    print("向量无穷范数", np.linalg.norm(a,ord=np.inf))

    a = np.array([[1.0, 3.0], [2.0, 1.0]])
    print("矩阵 F 范数", np.linalg.norm(a, ord="fro"))

    A = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0]])
    # 计算特征值
    print("特征值:", np.linalg.eigvals(A))
    # 计算特征值和特征向量
    eigvals, eigvectors = np.linalg.eig(A)
    print("特征值:", eigvals)
    print("特征向量:", eigvectors)

def ya2():
    X = np.hstack((np.array([[-0.5, -0.45, -0.35, -0.35, -0.1, 0, 0.2, 0.25, 0.3, 0.5]]).reshape(-1, 1), np.ones((10, 1)) * 1))
    print(X.T)
    y = np.array([-0.2, 0.1, -1.25, -1.2, 0, 0.5, -0.1, 0.2, 0.5, 1.2]).reshape(-1, 1)
    # 用公式求权重
    w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    hat_y = X.dot(w)
    #print("Weight:{}".format{list(w)})
    x = np.linspace(-1, 1, 50)
    hat_y = x * w[0] + w[1]
    plt.figure(figsize=(4, 4))
    plt.xlim(-1.0, 1.0)
    plt.xticks(np.linspace(-1.0, 1.0, 5))
    plt.ylim(-3, 3)
    plt.plot(x, hat_y, color='red')
    plt.scatter(X[:, 0], y[:, 0], color='black')
    plt.xlabel('$x_1$')
    plt.ylabel('$y$')
    plt.title('$Linear Regression$')
    plt.show()

def ya3():
    a = tf.zeros([3, 4])
    b = tf.ones([3,3])
    c = tf.fill([2, 2], 9)
    print("a:", a)
    print("b:", b)
    print("c:", c)
    print("***********************************")
    # 1.服从0.5为均值，1为方差的高斯分布数据生成一个2*3的tensor
    d = tf.random.normal([2, 3], mean=0.5, stddev=1)
    print("d:", d)
    #print(tf.function(d))
    # 2.在tf.truncated_normal中如果x的取值在区间（μ-2σ，μ+2σ）之外则重新进行选择。这样保证了生成的值都在均值附近。
    e = tf.random.truncated_normal([2, 3], mean=0.5, stddev=1)
    print("e:", e)
    # 3.返回2*2的矩阵，产生于-1和0之间，产生的值是均匀分布的。
    f = tf.random.uniform([2, 2], minval=-1, maxval=0)
    print("f:", f)
    print()

    a = tf.constant(3.)
    b = tf.constant(5.)
    print('a+b=', a + b)
    print()

    a = tf.constant([1, 5], dtype=tf.int64)
    print("a:", a)
    print("a.dtype:", a.dtype)
    print("a.shape:", a.shape)
    print()

    a = tf.Variable(tf.constant([1.0, 2.0, 3.0]))
    print("a:", a)
    print("a.dtype:", a.dtype)
    print("a.shape:", a.shape)
    print()

    a = np.arange(1, 6)
    # 1.将numpy中的array转为tensor
    b = tf.convert_to_tensor(a, dtype=tf.int64)
    print("a:", a, "type:", type(a))
    print("b:", b, "type:", b.dtype)
    print()

    x1 = tf.constant([1., 2., 3., 4.], dtype=tf.float64)
    print("x1:", x1)
    # 1.将值类型float64修改为int32
    x2 = tf.cast(x1, tf.int32)
    print("x2", x2)
    # 2.获取tensor中的最值
    print("minimum of x1：", tf.reduce_min(x1))
    print("maximum of x2：", tf.reduce_max(x2))
    print()

    x = tf.constant([[1, 2, 3], [2, 2, 3]])
    print("x:", x)
    # 1.求x中所有数的均值
    print("mean of x:", tf.reduce_mean(x))
    # 2.求每一行的和
    print("sum of x:", tf.reduce_sum(x, axis=1))
    print()

    a = tf.ones([1, 3])
    b = tf.fill([1, 3], 3.)
    print("a:", a)
    print("b:", b)
    print("a+b:", tf.add(a, b))
    print("a-b:", tf.subtract(a, b))
    print("a*b:", tf.multiply(a, b))
    print("b/a:", tf.divide(b, a))

    a = tf.fill([1, 2], 3.)
    print("a:", a)
    print("a的次方:", tf.pow(a, 3))
    print("a的平方:", tf.square(a))
    print("a的开方:", tf.sqrt(a))

def ya4():
    # tensorflow常用函数
    '''

    tf.cast(张量名, dtype=数据类型)  # 强制转换为该种数据类型
    tf.reduce_min(张量名)  # 计算张量维度上的最小值
    tf.reduce_max(张量名)  # 计算张量维度上的最大值
    tf.reduce_mean(张量名, axis=1)  # 求x轴方向上的平均值
    tf.reduce_sum(张量名, axis=0)  # 求y轴方向上的总和

    # tf.Variable()函数将变量标记为“可训练的”，被标记的变量会在反向传播中记录梯度信息，在神经网络训练中，常用来标记待训练参数
    tf.Variable(初始值)
    tf.Variable(tf.random_normal([2, 2], mean=0, stddev=1))

    # 四则运算（只有维度相同的张量才能做四則运算）
    tf.add()  # 加
    tf.subtract()  # 减
    tf.multiply()  # 乘
    tf.divide()  # 除
    tf.square()  # 平方
    tf.pow()  # 次方
    tf.sqrt()  # 开方
    tf.matmul()  # 矩阵乘法
    # 矩阵生成
    tf.ones()

    tf.fill()
    '''

    # enumerate函数
    # enumerate是python的内置函数，他可以遍历每个元素（如列表，与元组或字符串）输出他们的索引和元素常在循环中使用
    # 输出索引，元素
    seq = ["one", "two", "three"]
    for i, element in enumerate(seq):
        print(i, element)


    with tf.GradientTape() as tape:  # with结构记录计算过程，gradient求出张量的梯度
        w = tf.Variable(tf.constant(3.0))  # 初始化变量w
        loss = tf.pow(w, 2)  # 损失函数为w^2
    grad = tape.gradient(loss, w)  # 对w进行求导
    tf.print(grad)  # 输出结果

    # one-hot编码
    # tf.one_hot(待转换的数据，depth=分几类)
    labels = tf.constant([1, 0, 2])
    output = tf.one_hot(labels, 3)
    print(output)

    # softmax函数 tf.nn.softmax()使输出符合概率分布
    y = tf.constant([1.01, 2.01, -0.66])
    y_label = tf.nn.softmax(y)
    print(y_label)

    # assign_sub函数（更新经过tf.Variable定义过的变量）
    w = tf.Variable(4)
    w.assign_sub(1)  # 进行w=w-1操作
    print(w)

    # tf.argmax/tf.argmax返回张量沿指定最大值/最小值的索引tf.argmax(张量名，axis=方向)
    test = np.array([[1, 2, 3], [21, 8, 9], [9, 5, 4]])
    print(test)
    print(tf.argmax(test, axis=0))  # 输出y轴方向最大值的索引
    print(tf.argmax(test, axis=1))  # 输出x轴方向最小值的索引

def ya5():#one hot
    import pandas as pd
    df = pd.DataFrame([
        ['green', 'M', 10.1, 'class1'],
        ['red', 'L', 13.5, 'class2'],
        ['blue', 'XL', 15.3, 'class1']])

    df.columns = ['color', 'size', 'prize', 'class label']

    size_mapping = {
        'XL': 3,
        'L': 2,
        'M': 1}
    df['size'] = df['size'].map(size_mapping)

    class_mapping = {label: idx for idx, label in enumerate(set(df['class label']))}
    df['class label'] = df['class label'].map(class_mapping)

    print()
    from sklearn import preprocessing
    enc = preprocessing.OneHotEncoder()
    enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])    # fit来学习编码
    enc.transform([[0, 1, 3]]).toarray()    # 进行编码
if __name__ == '__main__':
    #ya1()
    #ya2()
    #ya3()
    #ya4()
    ya5()