import os
import cv2
import csv
import math


def crop(filepath,destpath):
    pathdir=os.listdir(filepath)
    for alldir in pathdir:
        child=os.path.join(filepath,alldir)  #被裁减图片的路径
        dest=os.path.join(destpath,alldir)   #裁减后图片的路径
        if os.path.isfile(child):
            image=cv2.imread(child)
            sp=image.shape
            sz1=int(sp[1])# width-x
            sz2=int(sp[0])# height-y
            #with open('./alter.csv','r') as f:
            with open("./train.csv",'r') as f:  #训练集标签的csv文件，文件中标签格式为：图片名称、4个点坐标，类别
                lines=csv.reader(f)
                for line in lines:
                    name=line[0]
                    if name==alldir:
                        name,x1,y1,x2,y2,x3,y3,x4,y4,x_cen,y_cen,cla=center(line)
           #通过center函数把文档中每行的内容提取出来
                        a,b,c,d=right_crop(x1,y1,x2,y4)
           #right_crop函数返回裁减后的各个相应位置
           #下边算出标志在裁剪后图片中的位置
                        x1=x1-a
                        y1=y1-c
                        x2=x3-a
                        y2=y3-c
           #在算法中我们需要的标签格式是归一化形式的x,y,w,h,round取结果的前六位
                        x_cen=round(((x2-x1)/2+x1)/416,6)
                        y_cen=round(((y2-y1)/2+y1)/416,6)
                        width=round((x2-x1)/416,6)
                        height=round((y2-y1)/416,6)
                        #with open('./4.txt','a') as f:
                            #f.write('%s %s %s %s %s\n'%(name,x_cen,y_cen,width,height))
                        na=name.split('.')[0]+'.txt'
                        na='/data2/ggp/MNist/flower/DF/label/test/%s'%(na)
                        #na='/home/hc/eriklindernoren/data/demo/label/test/%s'%(na)
                        write_label(na,x_cen,y_cen,width,height,cla) #把得到的标签写入文档

                        cropImg=image[c:d,a:b]     #裁剪图片
                        cv2.imwrite(dest,cropImg)  #保存图片

def right_crop(x1,y1,x2,y4):
#裁减的原图像为3200X1800，由于标志可能出现在图像中每个位置，因此我们要根据标志位置来裁剪，
#为保证裁剪图片不超出图片范围，比较标志左右两边和图片边界的距离，以近的那一边的4/5作为裁剪点，
#因为最终要得到416X416的图片大小，所以用刚刚得到的点+（或-）416，上下的裁剪范围同理
    if x1<3200-x2:
        a=math.floor(4/5*(x1))
        b=a+416
    else:
        b=math.floor(3200-4/5*(3200-x2))
        a=b-416
    if y1<1800-y4:
        c=math.floor(4/5*(y1))
        d=c+416
    else:
        d=math.floor(1800-4/5*(1800-y4))
        c=d-416
    return a,b,c,d



def write_label(na,x_cen,y_cen,width,height,cla):
    with open(na,'w') as f:
        f.write('%s %s %s %s %s'%(cla,x_cen,y_cen,width,height))
        f.write('\n')
        f.close()


def center(line):
    name=line[0]
    x1=int(line[1])
    y1=int(line[2])
    x2=int(line[3])
    y2=int(line[4])
    x3=int(line[5])
    y3=int(line[6])
    x4=int(line[7])
    y4=int(line[8])
    cla=line[9]
    x_cen=(x2-x1)/2+x1 
    y_cen=(y3-y2)/2+y2
    return name,x1,y1,x2,y2,x3,y3,x4,y4,x_cen,y_cen,cla


if __name__ == "__main__":
    crop('./images/train','./train')     #前者是原图路径，后者是裁剪图片存放路径
