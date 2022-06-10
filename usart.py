import serial
from time import sleep
j='*r080#'

def send(str):
    a = str+ "\n"
    # print(len(a))
    serial.write((a).encode("gbk"))
    sleep(0.1)
def receive():
    while(1):
        data = serial.read(10)
        t=data.decode("gbk").split("*")
        for i in t:
            if len(i)==3:
                return i[0:2]
        #if data != b'':
        #print("receive:", data.decode("gbk"))
        #return data.decode("gbk")
if __name__ == '__main__':
    serial = serial.Serial('COM1', 9600, timeout=2)  # /dev/ttyUSB0
    if serial.isOpen():
        print("open success")
    else:
        print("open failed")
    send("66666")
    start = 'ya'
    datal = 'l'
    datar = 'r'
    datau = 'u'
    datad = 'd'
    while True:
        if(receive() == start):
            print("ok")
            send(j)
            '''
            if (receive() == datal):
                print("ok2")
                jd=int(receive())
                print(jd)
            '''

