import numpy as np
a=np.empty([1,1],np.float)
print(a)
a=np.insert(arr=a,obj=a.shape[1],values=[2],axis=1)
print(a)
a=np.insert(arr=a,obj=a.shape[1],values=[3],axis=1)
print(a)
a=[[1,2,3]]
b=[[4,5,6]]
c=np.concatenate((a,b),axis=1)
#print(c)

