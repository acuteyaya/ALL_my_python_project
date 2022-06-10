import threading
n=0
m=0
def work1():
    global n,m
    while 1:
        threadLock.acquire()
        n=n+1
        print("n",n)
        threadLock.release()
def work2():
    global n,m
    while 1:
        threadLock.acquire()
        m = m + 1
        print("m",m)
        threadLock.release()

threadLock = threading.Lock()
threads = []
thread1=threading.Thread(target=work1)
thread2=threading.Thread(target=work2)
thread1.start()
thread2.start()
threads.append(thread1)
threads.append(thread2)
for t in threads:
    t.join()
print ("Exiting")
