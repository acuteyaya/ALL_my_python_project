def ya(n=0):
    if(n==0):
        for i in range(1,10):
            for j in range(1,i+1):
                print(f'{i}X{j}={i * j}',end=' ')
            print()
    else:
        for i in range(9,0,-1):
            for j in range(1, i + 1):
                print(f'{i}X{j}={i * j}', end=' ')
            print()
if __name__ == '__main__':
    ya()
    ya(1)
