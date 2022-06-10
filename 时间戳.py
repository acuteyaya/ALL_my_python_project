import time
count=0
sum=0
TIME = lambda:int(time.time() * 1000)
while(count<=10):
    timetemp1=TIME()
    time.sleep(1)
    timetemp2=TIME()
    timetemp=timetemp2-timetemp1
    sum=sum+timetemp
    print(sum)
    tttt=int(timetemp/1000)
    sttt=str(tttt)+'s'+" "+str(float(timetemp-tttt*1000))+'ms'
    count = count + 1
tttt=int(sum/1000)
sttt=str(tttt)+'s'+" "+str(float(sum-tttt*1000))+'ms'
print(sttt)
sum=0

