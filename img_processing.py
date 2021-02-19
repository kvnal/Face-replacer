import cv2 as cv
import numpy as np

img= cv.imread('img/faces.jpg')

def detectFace(img):
    face = cv.CascadeClassifier('haar_cascade/face.xml')
    Gimg =cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret =face.detectMultiScale(Gimg,1.3,4)
    return ret


def effects(img,type):
    if type.lower()=="blur":
       func=lambda img: cv.blur(img,(20,20)) 
    elif type.lower()=="moisac":
    #    func=lambda img: cv.resize(img,(16,16),interpolation=cv.INTER_LINEAR)
        print()
    elif type.lower()=="cartoon":
        def func(img):
            edge=cv.adaptiveThreshold(cv.cvtColor(img,cv.COLOR_BGR2GRAY),255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,5)
            edgeSmooth=cv.medianBlur(edge,5)
            return cv.bitwise_and(img,img,mask=edgeSmooth)
    elif type.lower()=="flip":
        func=lambda img:cv.flip(img,0)
    elif type.lower()=="pewdiepie":
        pie="img/pewdiepie.jpg"
        pieWall=cv.imread(pie)
        func=lambda img: cv.resize(pieWall,(img.shape[0],img.shape[1]))            
    elif type.lower()=="beauty":
        func=lambda img: cv.fastNlMeansDenoisingColored(img,None,7,7,5)
    elif type.lower()=="xi":
        eye=cv.CascadeClassifier("haar_cascade/eye.xml")
        char="X"
        # █ ▒ Ø Ǒ ♥
        color=(255,255,0)
        def func(img):
            Gimg=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            ret=eye.detectMultiScale(Gimg,1.2,2)
            for (x,y,w,h) in ret:
                cv.putText(img,char,(x,y+h),cv.FONT_HERSHEY_COMPLEX_SMALL,color=color,fontScale=2,thickness=3)
            return img
    else:
        print("unknown effect type!")
        return None

    ret=detectFace(img)
    for (x,y,h,w) in ret:
        img[y:y+h,x:x+w]=func(img[y:y+h,x:x+w])
    return img


def boxFace(img,BOX=True,OUTLINE=False):
    if not BOX and OUTLINE:
        print("Can't shape outline without value BOX=True\n\n")
        return None
    ret=detectFace(img)
    output=img.copy()

    len_ret=len(ret)-1
    for i,(x,y,w,h) in enumerate(ret):
        if OUTLINE:
            cv.rectangle(output,(x,y),(x+w,y+h),255,20)

        if i==0:
            x1,y1,w1,h1=x,y,w,h
            face=img[y:y+h,x:x+w]
            continue
            
        face=cv.resize(face,(w,h),None)

        if BOX:
            output[y:y+h,x:x+w]=face
        else:
            output=cv.seamlessClone(face,output,None,(x+h//2,y+h//2),cv.NORMAL_CLONE)
            
        face=img[y:y+h,x:x+w]
        
        if i==len_ret:
            face=cv.resize(face,(w1,h1),None)
            if BOX:
                output[y1:y1+h1,x1:x1+w1]=face
            else:
                output=cv.seamlessClone(face,output,None,(x1+h1//2,y1+h1//2),cv.NORMAL_CLONE)
            return output
    return None #
        



cv.imshow("",cv.resize(effects(img,"xi"),None,fx=1/3,fy=1/3))
cv.waitKey(0)
cv.destroyAllWindows()
