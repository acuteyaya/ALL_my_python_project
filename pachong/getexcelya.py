'''
File Name          : getexcel.py
Author             : CUTEYAYA
Version            : V1.0.0
Created on         : 2022/4/2
'''
import xlrd
import xlwt
import datetime
from PySide2.QtWidgets import QApplication
from PySide2.QtUiTools import QUiLoader

font = xlwt.Font()
# 字体类型
font.name = 'name Times New Roman'
# 字体颜色
font.colour_index = 0
# 字体大小，11为字号，20为衡量单位
#font.height = 20 * 11
font.height = 180
# 字体加粗
font.bold = 1
# 下划线
font.underline = 0
# 斜体字
font.italic = 0

# 设置单元格对齐方式
alignment = xlwt.Alignment()
# 0x01(左端对齐)、0x02(水平方向上居中对齐)、0x03(右端对齐)
alignment.horz = 0x02
# 0x00(上端对齐)、 0x01(垂直方向上居中对齐)、0x02(底端对齐)
alignment.vert = 0x01

# 设置自动换行
alignment.wrap = 1

# 设置边框
borders = xlwt.Borders()
# 细实线:1，小粗实线:2，细虚线:3，中细虚线:4，大粗实线:5，双线:6，细点虚线:7
# 大粗虚线:8，细点划线:9，粗点划线:10，细双点划线:11，粗双点划线:12，斜点划线:13
borders.left = 1
borders.right = 1
borders.top = 1
borders.bottom = 1
borders.left_colour = 0
borders.right_colour = 0
borders.top_colour = 0
borders.bottom_colour = 0

# 设置背景颜色
pattern = xlwt.Pattern()
# 设置背景颜色的模式
pattern.pattern = xlwt.Pattern.SOLID_PATTERN
# 背景颜色
pattern.pattern_fore_colour = 1

# 设置背景颜色
pattern1 = xlwt.Pattern()
# 设置背景颜色的模式
pattern1.pattern = xlwt.Pattern.SOLID_PATTERN
# 背景颜色
pattern1.pattern_fore_colour = 52

# 初始化样式
style0 = xlwt.XFStyle()
style0.font = font
style0.pattern = pattern#白色背景
style0.alignment = alignment
style0.borders = borders

style1 = xlwt.XFStyle()
style1.font = font
style1.pattern = pattern1#棕色背景
style1.alignment = alignment
style1.borders = borders
left=0
right=0
yaH=[]
yaI=[5]
yaJ=[]
yaL=[]
yaM=[7]
yaN=[]
yaAVERAGE3=[]
yaAVERAGE4=[]
yanew=[[]]
yaend=[[]]
yatime=["日期"]
ya1=["生产1"]
ya2=["生产2"]
ya3=["生产3"]
ya4=["生产4"]
ya10=[13]
ex1=["平均值"]
ex2=["显示成功率"]
ex3=["投入产出比"]
ex4=["结果"]
def yawxls(pathwrite):
    pathwrite1=pathwrite+"all.xls"
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('Sheet1', cell_overwrite_ok=True)
    for i in range(len(ex1)):
        worksheet.write(i, 0, ex1[i], style0)
        worksheet.write(i, 1, ex2[i], style0)
        worksheet.write(i, 2, ex3[i], style0)
        worksheet.write(i, 3, ex4[i], style0)
    workbook.save(pathwrite1)
def yawxls1(pathwrite,av):
    pathwrite1=pathwrite+str(av)+"a.xls"
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('Sheet1', cell_overwrite_ok=True)
    for i in range(len(yatime)):
         worksheet.write(i, 1, yatime[i], style0)

    worksheet.write(0, 2, ya1[0], style0)
    for i in range(1,len(ya1)):
         worksheet.write(i, 2, ya1[i], style1)

    worksheet.write(0, 3, ya2[0], style0)
    for i in range(1, len(ya2)):
        worksheet.write(i, 3, ya2[i], style1)

    worksheet.write(0, 4, ya3[0], style0)
    for i in range(1, len(ya3)):
        worksheet.write(i, 4, ya3[i], style1)

    worksheet.write(0, 5, ya4[0], style0)
    for i in range(1, len(ya4)):
        worksheet.write(i, 5, ya4[i], style1)
    ########################################H
    yalen=len(yaH)

    for i in range(0, yalen):
        worksheet.write(av+1+i, 6, yaAVERAGE3[i], style0)
    for i in range(0, yalen):
        worksheet.write(av+1+i, 7, yaH[i], style0)
    for i in range(0, yalen):
        worksheet.write(av+1+i, 8, yaI[i], style0)
    for i in range(0, yalen):
        worksheet.write(av+1+i, 9, yaJ[i], style0)
    for i in range(0, yalen):
        worksheet.write(av+1+i, 10, yaAVERAGE4[i], style0)
    for i in range(0, yalen):
        worksheet.write(av+1+i, 11, yaL[i], style0)
    for i in range(0, yalen):
        worksheet.write(av+1+i, 12, yaM[i], style0)
    for i in range(0, yalen):
        worksheet.write(av+1+i, 13, yaN[i], style0)
    #worksheet.write_merge(10, 10 + 20, 7, 7)
    workbook.save(pathwrite1)
    yawxls2(pathwrite,av)

def yawxls2(pathwrite,av):
    global yanew
    #print(yanew)
    pathwrite1 = pathwrite+str(av) + "a.xls"
    pathwrite2 = pathwrite+str(av) + "b.xls"
    yatemp=["日期","生产1","生产2","生产3","生产4"]
    data = xlrd.open_workbook(pathwrite1)
    table = data.sheet_by_name(u'Sheet1')
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('Sheet1', cell_overwrite_ok=True)
    temp=0
    for i in range(av+1,table.nrows):
        l = table.row_values(i)
        if(int(l[9])==8 or int(l[13])==10):
            yanew.append(l)

            if(int(l[9])==8):
                if(temp!=0):
                    if (int(l[9])==temp):
                        ya10.append(12)
                    else:
                        ya10.append(13)
                temp=8
            else:
                if (temp != 0):
                    if (int(l[13])==temp):
                        ya10.append(12)
                    else:
                        ya10.append(13)
                temp=10
    for i in range(0, 5):
        worksheet.write(0, 1+i,  yatemp[i], style0)
    for i in range(1, len(yanew)):
        for j in range(len(yanew[i])):
            worksheet.write(i, j, yanew[i][j],style0)
    for i in range(0, len(ya10)):
        worksheet.write(i+1, 14, ya10[i], style0)
    workbook.save(pathwrite2)
    yawxls3(pathwrite,av)

def yawxls3(pathwrite,av):
    pathwrite1 = pathwrite+str(av) + "a.xls"
    pathwrite2 = pathwrite+str(av) + "b.xls"
    pathwrite3 = pathwrite+str(av) + "c.xls"
    yatemp = ["日期", "生产1", "生产2", "生产3", "生产4","","H列","成功率","投入产出比"]
    data = xlrd.open_workbook(pathwrite2)
    table = data.sheet_by_name(u'Sheet1')
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('Sheet1', cell_overwrite_ok=True)
    for i in range(1,table.nrows):
        l = table.row_values(i)
        if(int(l[14])==13):
            temp = []
            for j in range(1, 6):
                temp.append(l[j])
            temp.append("")
            temp.append(l[7])
            #temp.append(l[14])
            yaend.append(temp)
    for i in range(0, 9):
        worksheet.write(0, i,  yatemp[i], style0)
    for i in range(1, len(yaend)):
        for j in range(len(yaend[i])):
            worksheet.write(i, j, yaend[i][j], style0)
    workbook.save(pathwrite3)
    data = xlrd.open_workbook(pathwrite3)
    table = data.sheet_by_name(u'Sheet1')
    tag=0
    for i in range(1,table.nrows-1):#0结尾
        l = table.row_values(i)
        if(l[6]==0):
            tag=1
            continue
        else:
            j=0
            p=0
            k1=0
            k2=0
            wz=10000
            for j in range(i, table.nrows-1,2):
                lj1 = table.row_values(j)
                lj2 = table.row_values(j+1)
                t=(float(lj2[1])-float(lj1[1]))/float(lj1[1])
                t=t*100
                k1=k1+1
                if(t>=0):
                    k2=k2+1
                p=((wz/float(lj1[1]))*float(lj2[1])-10000)/10000
                wz=wz/float(lj1[1])*float(lj2[1])
                p = p * 100
               #t=(lj2[1]-lj1[1])/lj1[1]
                worksheet.write(j + 1, 7, str('{:.3f}'.format(t))+"%", style0)
                worksheet.write(j + 1, 8, str('{:.3f}'.format(p)) + "%", style0)
                worksheet.write(j + 1, 9, str('{:.3f}'.format(wz)) , style0)
            t=k2/k1
            t = t * 100
            if(tag==1):
                worksheet.write(j + 3, 7, "成功率=" + str(k2) + "/" + str(k1) + "=" + str('{:.3f}'.format(t)) + "%", style0)
                worksheet.write(j + 3, 8, "投入产出比=" + str('{:.3f}'.format(p)) + "%", style0)
            else:
                worksheet.write(j + 2, 7, "成功率="+str(k2)+"/"+str(k1)+"="+str('{:.3f}'.format(t)) + "%", style0)
                worksheet.write(j + 2, 8, "投入产出比="+str('{:.3f}'.format(p)) + "%", style0)
            ex1.append(av)
            ex2.append(t)
            ex3.append(p)
            if(t>=0 and p>=0):
                ex4.append("成功")
            else:
                ex4.append("失败")
            break

    workbook.save(pathwrite3)

def yarxls(pathread,pathwrite):
    pathread=pathread+".xls"
    global yatime,ya1,ya2,ya3,ya4,yaAVERAGE3,yaAVERAGE4,ya10
    global yaH,yaI,yaJ,yaL,yaM,yaN,yanew,yaend
    data = xlrd.open_workbook(pathread)
    table = data.sheet_by_name(u'Sheet1')
    #print(table.nrows)  # 输出表格行数
    #print(table.ncols)  # 输出表格列数
    for i in range(1,table.nrows):
        l = table.cell_value(i,0)
        datets = datetime.datetime(*xlrd.xldate_as_tuple(l, 0))
        cell = datets.strftime('%Y/%m/%d')
        yatime.append(cell)
    for i in range(1,table.nrows):
        l = table.cell_value(i,1)
        ya1.append(l)
    for i in range(1,table.nrows):
        l = table.cell_value(i,2)
        ya2.append(l)
    for i in range(1,table.nrows):
        l = table.cell_value(i,3)
        ya3.append(l)
    for i in range(1,table.nrows):
        l = table.cell_value(i,4)
        ya4.append(l)

    yalen=len(ya1)
  #  print(yalen)
    for av in range(left,right+1,1):
        yaH = []
        yaI = [5]
        yaJ = []
        yaL = []
        yaM = [7]
        yaN = []
        yaAVERAGE3 = []
        yaAVERAGE4 = []
        yanew = [[]]
        yaend = [[]]
        ya10 = [13]
        for i in range(1,yalen-av):
            sum3 = 0
            sum4 = 0
           # print(ya3)
            for j in range(i,i+av):
                sum3 += float(ya3[j])
                sum4 += float(ya4[j])
            AVERAGE3 = sum3 / av
            AVERAGE4 = sum4 / av
            yaAVERAGE3.append(AVERAGE3)
            yaAVERAGE4.append(AVERAGE4)
            if (float(ya1[i + av]) > AVERAGE3):
                yaH.append(1)
            else:
                yaH.append(0)
            if (float(ya1[i + av]) < AVERAGE4):
                yaL.append(2)
            else:
                yaL.append(3)
        #1
        for i in range(1,yalen-av-1):
           if(yaH[i]==yaH[i-1]):
               yaI.append(4)
           else:
               yaI.append(5)
        for i in range(0,yalen-av-1):
           if(yaH[i]==1 and yaI[i]==5):
               yaJ.append(8)
           else:
               yaJ.append(9)
        #2
        for i in range(1,yalen-av-1):
           if(yaL[i]==yaL[i-1]):
               yaM.append(6)
           else:
               yaM.append(7)
        for i in range(0,yalen-av-1):
           if(yaL[i]==2 and yaM[i]==7):
               yaN.append(10)
           else:
               yaN.append(11)
        yawxls1(pathwrite,av)
class Stats:
    def __init__(self):
        self.ui = QUiLoader().load("./ya.ui")
        self.ui.pushButton_2.clicked.connect(self.handleCalc2)
    def handleCalc2(self):
        pathread = self.ui.lineEdit.text()  # 读
        pathwrite = self.ui.lineEdit_2.text()  # 写
        global left,right
        left=int(self.ui.lineEdit_3.text())
        right=int(self.ui.lineEdit_4.text())
        yarxls(pathread,pathwrite)
        yawxls(pathwrite)
        self.ui.textEdit.setPlaceholderText("已生成,请勿再次生成")

if __name__ == '__main__':
    app = QApplication([])
    stats = Stats()
    stats.ui.show()
    app.exec_()
