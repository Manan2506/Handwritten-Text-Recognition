import cv2
import statistics as st
import numpy as np
from word_segment import prepareImg
from word_segment import wordSegmentation
from infertext import infer
import tensorflow as tf
import os.path
import glob

img_dir = r"./Buffalo_Dataset" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
directory="Buffalo_TEXT"
path=os.path.join('./',directory)
os.mkdir(path)
save_path= './'+directory
#int(input())
for f1 in files:
    img = cv2.imread(f1,cv2.IMREAD_GRAYSCALE)
    print(img.shape)
    img = np.array(img, dtype=np.uint8)
    #print(f1)
    i=340
    mi=st.mean(img[i][60:100])
    x=0
    for j in range(30):
        up=st.mean(img[i+j][60:100])
        #print(up,mi)
        if up<mi:
            mi=up
            x=j
            
        li=i+x
    #print(li)
    ret, thresh= cv2.threshold(img, 105, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)

    #img = cv2.dilate(thresh, kernel, iterations=1) 

    img = cv2.erode(thresh, kernel, iterations=1)
    cv2.imwrite("re.jpg", img)
    #print(f1.type)
    
    completeName = os.path.join(save_path, f1[-16:-4]+".txt")
    f=open(completeName,"w+")
    #for i in list_of_files:
    img1=img[:li+1]
    res = wordSegmentation(img1, kernelSize=101, sigma=11, theta=10, minArea=1000)
    te=[]
    l=np.full((100,1),255)
    temp=np.full((100,100),255)
    for (j, w) in enumerate(res):
        (wordBox, wordImg) = w
             #cv2.imwrite("image3.jpg",wordImg)
        (x,y,w1,h)=wordBox
        if y+h//2 > 40:
            hi=100
            dim=(wordImg.shape[1],hi)
            resi= cv2.resize(wordImg, dim, interpolation = cv2.INTER_AREA)
            te+=[w[1]]
                 #l=temp
            l=np.append(l,resi,axis=1)
            l=np.append(l,temp,axis=1)
                 #print(l.shape)
        
        #cv2.imwrite("image3.jpg",l)
        #int(input("from2"))
        if len(l[0])>1:
            text=infer(te)
            f.write(" ".join(str(item) for item in text[0]))
            f.write('\n')
    #input()
    res=[]
    a=np.array([[]])
    a=[]
    temp=np.full((100,100),255)
    for i in range(li,3150,103):
        if (sum(img[i+15][800:1600])/800+sum(img[i+16][800:1600])/800+sum(img[i+17][800:1600])/800)/3<=240:
            img1=img[i+3:i+106]        
            res = wordSegmentation(img1, kernelSize=101, sigma=11, theta=10, minArea=7000)    
        elif st.mean(img[i+90][240:2333])>=253:
            f.write('\n')
            continue
        else:
            img1=img[i+20:i+150]
            res = wordSegmentation(img1, kernelSize=101, sigma=11, theta=10, minArea=1000)
        
        cv2.imwrite("img1.jpg",img1)
        #int(input("from1"))
        te=[]
        l=np.full((100,1),255)
        for (j, w) in enumerate(res):
             (wordBox, wordImg) = w
             #cv2.imwrite("image3.jpg",wordImg)
             (x,y,w1,h)=wordBox
             if y+h//2 > 40:
                 hi=100
                 dim=(wordImg.shape[1],hi)
                 resi= cv2.resize(wordImg, dim, interpolation = cv2.INTER_AREA)
                 te+=[w[1]]
                 #l=temp
                 l=np.append(l,resi,axis=1)
                 l=np.append(l,temp,axis=1)
                 #print(l.shape)
        a+=[l]
        #cv2.imwrite("image3.jpg",l)
        #int(input("from2"))
        #print(l)
        if len(l[0])>1:
            text=infer(te)
            f.write(" ".join(str(item) for item in text[0]))
            f.write('\n')
            #print(text)
