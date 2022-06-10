import wordcloud
import jieba
from imageio import imread
mask = imread("1.jpg")#图案照片
f = open("wenben.txt", "r", encoding="utf-8")
t = f.read()
f.close()
t = jieba.lcut(t)
stopwords=['需要','主要','各类','晶圆','包括','等等','进行','很多','完成','非常','分为','相关']#有不要显示的文字加进去
counts= {}
for i in t:
    if len(i)>1:
        counts[i] = counts.get(i,0)+1
for word in stopwords:
    counts.pop(word,0)
l = sorted(counts.items(),key=lambda x:x[1],reverse=True)
k=0
yal=[]
for i in l:
    k=k+1
    if(k>=50):#显示五十个词
        break
    yal.append(i[0])
ya=['可爱鸭','张可爱','大可爱','可爱zcl','好可爱',"可爱卡比"]
yaya=ya*5
yayaya=[]
k=0
for i in yaya:
    temp=i

    temp=temp+str(k)

    yayaya.append(temp)
    k=k+1

print(yayaya) #要显示的词
w = wordcloud.WordCloud(width=1000,font_path="msyh.ttc",height=1000,background_color="white",mask=mask)
w.generate(" ".join(yayaya))
w.to_file("CUTEYAYA.png")#生成图片路径 需要的话俺可以帮你自动跳出来

