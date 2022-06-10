#初始状态，假设杯子都没水（0，0）
#期望状态，（60，-），（-，60）
#这里需要知道每一步后续操作是什么
#先定义一个函数来表示状态以及对应的操作
#(x,y)表示当前状态，X,Y表示容器最大容量

def success(x,y,X,Y):
    return{
        (0,y):'倒空x',
        (x,0):'倒空y',
        (x+y-Y,Y) if x+y>=Y else (0,x+y):'x倒入y',
        (X,x+y-X) if x+y>=X else (x+y,0):'y倒入x',
        (X,y):'装满X',
        (x,Y):'装满Y'
    }
#if __name__=='__main__': #入口函数
 #   print(success(0,0,90,40))
#goal=60
#capacity1,capacity2=90,40#定义最大容量
#paths=[[('init',(0,0))]]#假设路径初始状态，路径用列表的形式表示
#现在需要不断地看这个paths，不断地搜索这条路的边沿，再沿每条路继续往下扩展
#search_solution()函数用来搜索路径的

path=[]
path1 = []
path2 = []
def search_solution(capacity1,capacity2,goal,start=(0,0)):
    path.append([('init', start)])
    explored=set()#为了避免死循环，设置的变量
    while path:
        #path=paths.pop(0) #当发现还有路径可探索时，这里任取一路径
        frontier=path[-1]#取路径的边沿，path的最右边
        (x,y)=frontier[-1]#(x,y)就是这个边沿状态
        x1,y1=y
        path.pop()
        ya=[]
        #for循环做个遍历，items()返回字典里的键、值
        for state,action in success(x1,y1,capacity1,capacity2).items():
           # print(frontier,state,action)

            if state in explored:
                continue #如果状态已经存在，则直接跳出此次循环，避免死循环
            new_path=[(action,state)]
            #new_path.append#假设状态尚不存在，把此时此刻的状态添加到路径中
            if goal in state:
                return new_path
            else:
                ya=new_path
                path1.append(ya)
                path.append(new_path)
                explored.add(state) #只保存状态，因为只要判断状态就饿可以，不需要判断操作

    return None
#print(search_solution(capacity1,capacity2,goal,(90,40)))
if __name__=='__main__':
    print(search_solution(90,40,60,(0,0)))
    print(path1)
