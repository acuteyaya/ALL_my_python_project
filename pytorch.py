import torch
import os
import pandas as pd
def yatorch():
    x=torch.arange(24)
    x=x.reshape(4,6)
    x[0:2,:]=666
    print(x,x[-1],x[1:3])
    y=torch.zeros(2)
    z1=torch.tensor([[1],[2]])
    #+ - * / **
    z2=torch.tensor([[1],[2]])
    z=torch.cat((z1,z2),dim=1)
    zsum=z.sum()#广播机制
    print(zsum)
def yaos():
    pass
if __name__ == '__main__':
    yatorch()