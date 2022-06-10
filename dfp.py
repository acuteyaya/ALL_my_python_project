from pynput import keyboard
maplist=[[0,1,0,1,0],
         [0,0,0,0,0],
         [0,0,0,0,0],
         [0,0,0,0,0],
         [0,0,0,0,0],
         [0,0,0,0,0]]
def listlen(l):
    max = 0
    for i, element in enumerate(l):
        if (len(element) > max):
            max = len(element)
    return i+1,max
mapsize=listlen(maplist)
h=int(mapsize[0])
w=int(mapsize[1])
x0=-1
y0=-1
def dfp(x,y):
    global maplist,x0,y0
    if(x==h or y==w or x<0 or y<0 or maplist[x][y]==1):
        return False
    if(maplist[x][y]==0):
        maplist[x][y]=8
        #print(x,y,x0,y0)
        if(x0>=0 and y0>=0):
            maplist[x0][y0]=0
        x0=x
        y0=y
    return True
def on_press(key):
    global x0,y0
    if (key.char=='s'):
        r = dfp(x0 + 1, y0)
    elif (key.char=='w'):
        r = dfp(x0 - 1, y0)
    elif (key.char == 'd'):
        r = dfp(x0, y0 + 1)
    elif (key.char == 'a'):
        r = dfp(x0, y0 - 1)
    if(r==False):
        print('damie')
def on_release(key):
    print()
    for i in maplist:
        print(i)
    if(x0==h-1 and y0 ==w-1):
        print("6666\n6666")
    if key == keyboard.Key.esc:
        return False
if __name__=="__main__":
    dfp(0,0)
    for i in maplist:
        print(i)
    print('start')
    with keyboard.Listener(
            on_press=on_press,
            on_release=on_release) as listener:
        listener.join()