class A(object):
    # 属性默认为类属性（可以给直接被类本身调用）
    num = "类属性"

    # 实例化方法（必须实例化类之后才能被调用）
    def func1(self):  # self : 表示实例化类后的地址id
        print("func1")
        print(self)

    # 类方法（不需要实例化类就可以被类本身调用）
    @classmethod
    def func2(cls):  # cls : 表示没用被实例化的类本身
        print("func2")
        print(cls)
        print(cls.num)
        cls().func1()

    # 不传递传递默认self参数的方法（该方法也是可以直接被类调用的，但是这样做不标准）
    def func3():
        print("func3")
        print(A.num)  # 属性是可以直接用类本身调用的
A1=A()
A1.func1() #这样调用是会D报错：因为func1()调用时需要默认传递实例化类后的地址id参数，如果不实例化类是无法调用的
A.func2()
A.func3()

class C(object):
    @staticmethod
    def f():
        print('runoob')

    def __del__(self):  # 定义析构函数
        print("del.....run...")
C.f();  # 静态方法无需实例化
cobj = C()
cobj.f()  # 也可以实例化后调用
del cobj

# 上下文管理器类
class TestWith(object):
    def __init__(self):
        pass

    def __enter__(self):
        """进入with语句的时候被调用
           并将返回值赋给as语句的变量名
        """
        print('__enter__')
        return "var"

    def __exit__(self, exc_type, exc_val, exc_tb):
        """离开with的时候被with调用"""
        print('__exit__')
        return True
# with后面必须跟一个上下文管理器
# 如果使用了as，则是把上下文管理器的 __enter__() 方法的返回值赋值给 target
# target 可以是单个变量，或者由“()”括起来的元组（不能是仅仅由“,”分隔的变量列表，必须加“()”）
with TestWith() as var:
    print(var)

class A:
    pass
class B(A):
    pass
isinstance(A(), A)  # returns True
type(A()) == A  # returns True
isinstance(B(), A)  # returns True
type(B()) == A  # returns False
#type()#不会认为子类是一种父类类型，不考虑继承关系。
#isinstance(object,classinfo)#会认为子类是一种父类类型，考虑继承关系。
# object -- 实例对象 classinfo -- 可以是直接或间接类名、基本类型或者由它们组成的元组

class A:
    def add(self, x):
        y = x + 1
        print(y)
class B(A):
    def add(self, x):
        super().add(x)# 用来解决多重继承问题的，直接用类名调用父类方法在使用单继承的时候没问题，
                      # 但是如果使用多继承，会涉及到查找顺序（MRO）、重复调用（钻石继承）等种种问题。
                      # MRO 就是类的方法解析顺序表, 其实也就是继承父类方法时的顺序表。
b = B()
b.add(2)  # 3


class FooParent(object):
    def __init__(self):
        self.parent = 'I\'m the parent.'
        print('Parent')
    def bar(self, message):
        print("%s from Parent" % message)
class FooChild(FooParent):
    def __init__(self):
        # super(FooChild,self) 首先找到 FooChild 的父类（就是类 FooParent），然后把类 FooChild 的对象转换为类 FooParent 的对象
        super(FooChild, self).__init__()
        print('Child')
    def bar(self, message):
        super(FooChild, self).bar(message)
        print('Child bar fuction')
        print(self.parent)
fooChild = FooChild()
fooChild.bar('HelloWorld')
