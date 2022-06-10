'''
import  socket    #socket模块
HOST ='192.168.31.239'
PORT = 520
s =  socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.bind((HOST,PORT))
s.listen(1)
conn,addr = s.accept()    #接受连接
while  1:
    data = conn.recv(1024)
    print(data)
    conn.sendall(b'999')
conn.close()

import  socket
HOST ='192.168.31.239'
PORT = 520
s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.connect((HOST,PORT))
while  1 :
    s.sendall(b"666")
    data = s.recv(1024)
    print(data)
s.close()
'''
import  socket
HOST ='192.168.31.239'
PORT = 520
jd='jd'
st='st'
bu='bu'
ud='ud'
lr='lr'
def yaface(yas):
    if(yas[0:2]==ud):
        jd=int(yas[2:5])
    elif(yas[0:2]==lr):
        jd = int(yas[2:5])
    print(jd)
def jdstbu(yas):
    if(yas==jd):
        pass
    elif(yas==st):
        pass
    elif(yas==bu):
        pass
if __name__ == '__main__':
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    s.connect((HOST,PORT))
    while  1 :
        data = s.recv(1024)
        data = data.decode()  # 第一参数默认utf8，第二参数默认strict
        if(data!=jd and data!=st and data!=bu):
            yaface(data)
        else:
            jdstbu(data)
    s.close()