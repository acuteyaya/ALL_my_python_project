l=[[1,1],[1],[1]]
def listlen(l):
    max = 0
    for i, element in enumerate(l):
        if (len(element) > max):
            max = len(element)
    return i+1,max
print(listlen(l))