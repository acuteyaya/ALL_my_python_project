import cv2
import os
import shutil
import sys
outx=350
outy=640
syspath=r'E:\window\tiaoshi\pycharm\yoloV5\yolov5-master\yamake\labelImg-master\zcl'
sys.path.append(syspath)
pathtemp=r'E:\window\tiaoshi\pycharm\yoloV5\yolov5-master\yamake\labelImg-master\zcl\mp4'
path=pathtemp.rsplit('\\',1)[0]
if not os.path.exists(path + r'\Change_Save_Dir'):
    os.makedirs(path + r'\Change_Save_Dir')
savepath=path+r'\img'
templist=[]
ya=[]
tag=0
if not os.path.exists(path + r'\tag.txt'):
    fp=open(path+r"\tag.txt","w")
    print("CREAT")
    fp.write("0")
    fp.close()
if(0):
    fp = open(path + r"\tag.txt", "w")
    print("tagok")
    fp.write("0")
    fp.close()
fp = open(path + r"\tag.txt", "r")
m=int(fp.read())
fp.close()
s=''
def read_path(path_name):
    for dir_item in os.listdir(path_name):
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        if os.path.isdir(full_path):
            read_path(full_path)
        else:
            if full_path.endswith('.mp4') or full_path.endswith('.MP4'):
                templist.append(full_path)
def yaimg():
    global m
    #print(pathtemp)
    read_path(pathtemp)
    #print(templist)
    if not os.path.exists(savepath + r'\Open_Dir'):
        os.makedirs(savepath + r'\Open_Dir')
    for i in templist:
        s=i.rsplit('\\',2)
        stemp=s[-2]
        s=savepath+'\\'+s[-2]
        yaya = s + "\\" + stemp + r".txt"
        if not os.path.exists(s + "\\"+stemp+r".txt"):
            print("creat")
            os.makedirs(s)
            fp = open(yaya,"w")
            fp.write("0")
            fp.close()
        fp = open(yaya,"r")
        n = int(fp.read())
        fp.close()
        if not os.path.exists(s):
            os.makedirs(s)
        cap = cv2.VideoCapture(i)
        k=0
        while cap.isOpened():
            ok, frame = cap.read()
            if( not ok):
                break
            if(k==2):
                k=0
                kn = (format(n, '0>5d'))
                km = (format(m, '0>5d'))
                #print(type(k))
                img_namen = r'%s\%s.jpg' % (s, kn)
                img_namem = r'%s\Open_Dir\%s.jpg' % (savepath, km)
                #print(s,frame)
                m=m+1
                n=n+1
                frame = cv2.resize(frame,dsize=(outx,outy),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(img_namen, frame)
                cv2.imwrite(img_namem, frame)
                #cv2.imshow("ya",frame)
            k=k+1
            cv2.waitKey(0)
        cap.release()
        print(i,"ok")
        fp = open(yaya,"w")
        fp.write(str(n))
        fp.close()
if __name__ == '__main__':
    yaimg()
    print("over")
    fp = open(path + r"\tag.txt", "w")
    fp.write(str(m))
    fp.close()
    print("all",m)
    shutil.rmtree(pathtemp)
    os.makedirs(pathtemp)