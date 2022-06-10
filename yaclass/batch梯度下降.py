import random
'''
loss=((wx+b)-y)**2
'''
def loss(x,w,b,y):
    return ((w*x+b)-y)**2

def gradient(w,x,b,y):
    #求w偏导
    return 2*(w*x+b-y)*x

w,b=random.randint(-10,10),random.randint(-10,10)

#x,y=0.2,0.35
x,y=10,0.35
#print(loss(x,w,b,y))
lr=1e-3
for i in range(100):
    w_gradent=gradient(w,x,b,y)
    w=w-w_gradent*lr
    #print('w=',w)
    print(loss(x,w,b,y))
