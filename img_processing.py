import cv2 as cv
import numpy as np


img= cv.imread('img/faces.jpg')

def detectFace(img):
    face = cv.CascadeClassifier('haar_cascade/face.xml')
    Gimg =cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret =face.detectMultiScale(Gimg,1.3,4)
    return ret


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
        



# cv.imshow('ret',ret)
out=boxFace(img,OUTLINE=True)
cv.imshow("",out)
cv.waitKey(0)
cv.destroyAllWindows()
