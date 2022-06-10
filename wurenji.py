import threading
import socket
import time
count = 0
host = ''
port = 9000
locaddr = (host,port)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
tello_address1 = ('192.168.11.76', 8889)
tello_address2 = ('192.168.11.81', 8889)
sock.bind(locaddr)
def sent1():

                msg = "command"
                msg = msg.encode(encoding="utf-8")
                sock.sendto(msg, tello_address1)
                sock.sendto(msg, tello_address2)
                time.sleep(5)

                msg = "takeoff"
                msg = msg.encode(encoding="utf-8")
                sock.sendto(msg, tello_address1)
                sock.sendto(msg, tello_address2)
                time.sleep(8)

                msg = "down 40"
                msg = msg.encode(encoding="utf-8")
                sock.sendto(msg, tello_address1)
                sock.sendto(msg, tello_address2)
                time.sleep(8)

                msg = "forward 400"
                msg = msg.encode(encoding="utf-8")
                sock.sendto(msg, tello_address1)
                msg = "forward 300"
                msg = msg.encode(encoding="utf-8")
                sock.sendto(msg, tello_address2)
                time.sleep(10)

                msg = "land"
                msg = msg.encode(encoding="utf-8")
                sock.sendto(msg, tello_address1)
                sock.sendto(msg, tello_address2)
def sent2():
        return 0
        try:
                msg = "command"
                msg = msg.encode(encoding="utf-8")
                sock.sendto(msg, tello_address2)
                time.sleep(5)

                msg = "takeoff"
                msg = msg.encode(encoding="utf-8")
                sock.sendto(msg, tello_address2)
                time.sleep(5)

                msg = "up 50"
                msg = msg.encode(encoding="utf-8")
                sock.sendto(msg, tello_address2)
                time.sleep(5)
                if (0):
                        msg = "forward 300"
                        msg = msg.encode(encoding="utf-8")
                        sock.sendto(msg, tello_address2)
                        time.sleep(20)

                msg = "land"
                msg = msg.encode(encoding="utf-8")
                sock.sendto(msg, tello_address2)

        except KeyboardInterrupt:
                print('\n 无人机2 已经寄了\n')
def recv():
    global count
    count=count+1

    while True:
        try:
            data, server = sock.recvfrom(1518)
            print(data.decode(encoding="utf-8"))
        except Exception:
            print ('\nExit . . .\n')
            break
if __name__ == '__main__':
        threads = []
        recvThread1 = threading.Thread(target=recv)
        sent1()
        #recvThread3 = threading.Thread(target=sent2)
        recvThread1.start()

        #recvThread3.start()
        threads.append(recvThread1)

        #threads.append(recvThread3)
        sock.close()
        for t in threads:
                t.join()
        print("Exiting")
