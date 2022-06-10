import operator

if __name__ == '__main__':
    print( "abs(-45) : ", abs(-45))

    divmod(7, 2)
    divmod(8, 2)#商和余数的元组

    print("{1} {0} {1}".format("hello", "world"))  # 设置指定位置
    print("{} 对应的位置是 {{0}}".format("runoob"))
    print("{:.2f}".format(3.1415926))

    a = ["he", "l", "l", "o"]
    print(" ".join(a))

    class Names():
        obj = 'world'
        name = 'python'
    print('hello {names.obj} i am {names.name}'.format(names=Names))

    with open('内置函数.txt', 'w') as f:
        data = 'some data to be written to the file'
        f.write(data)
        f.write('\n')
    with open('内置函数.txt', 'r') as f:
        data = f.read()
        f.seek(0,0)#(偏移量, 起始位置) 0：⽂件开头 1：当前位置 2：⽂件结尾
        yatell=f.tell()
        print("tell is %d"%yatell)

        #f.readline()
        print('{}'.format(data))

    all(['a', 'b', 'c', 'd'])  #列表元素都不为空或0
    all(('a', 'b', '', 'd'))  #元组存在一个为空的元素
    any(['a', 'b', 'c', 'd'])  # 列表list，元素都不为空或0
    any(['a', 'b', '', 'd'])

    i = 0
    seq = ['one', 'two', 'three']
    for element in seq:
        print(i, seq[i])
        i += 1
    for i, element in enumerate(seq):
        print(i, element)

    print(ord('a'))
    print(chr(48), chr(49), chr(97))
    
    x = 7
    eval('3 * x')
    eval('pow(2,2)')

    complex(1, 2)

    myslice = slice(5)  # 设置截取5个元素的切片
    #class slice(stop)
    #class slice(start, stop[, step]) start -- 起始位置 stop -- 结束位置 step -- 间距
    arr = range(10)
    print(arr[myslice])

    x = set('runoob')
    y = set('google')
    x & y  # 交集
    x | y  # 并集
    x - y  # 差集

    a = [1, 2, 3]
    b = [4, 5, 6]
    c = [4, 5, 6, 7, 8]
    zipped = zip(a, b)

    operator.eq('hello', 'name')

    hex(255)

    a = [5, 7, 6, 3, 4, 1, 2]
    b = sorted(a,reverse=True)
    print(b)

    print(round(123.45))

    it = iter([1, 2, 3, 4, 5])
    # 循环:
    while True:
        try:
            # 获得下一个值:
            x = next(it)
            print(x)
        except StopIteration:
            # 遇到StopIteration就退出循环
            break

    m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    n = [[2, 2, 2], [3, 3, 3], [4, 4, 4]]

    print('list(zip(m,n)):\n', list(zip(m, n)))
    print("*zip(m, n):\n", *zip(m, n))
    print("*zip(*zip(m, n)):\n", *zip(*zip(m, n)))

    m2, n2 = zip(*zip(m, n))
    print(m == list(m2) and n == list(n2))
