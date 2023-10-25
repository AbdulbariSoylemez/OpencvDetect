import cv2
import matplotlib.pyplot as plt
import numpy as np

## Fotohrftan yüz tespiti 
eğlen = cv2.imread("faceDetect/img/eğlen.JPG",0)
eğlen=cv2.cvtColor(eğlen,cv2.COLOR_BGR2RGB)




plt.figure(), plt.imshow(eğlen), plt.title("Eğlen")

yüz_caasced= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

yüz_rect=yüz_caasced.detectMultiScale(eğlen)

for (x,y,w,h) in yüz_rect:
     cv2.rectangle(eğlen,(x,y),(x+w,y+h),(255,255,0),10)

plt.figure(), plt.imshow(eğlen), plt.title("Eğlenceden yüz tespiti ")

## KAMERADAN YÜZ TESPİTİ

video = cv2.VideoCapture(0)


while True:

     ret, frame = video.read()
     if not ret:
        break
    
     yüz_caasced= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
     yüz_rect = yüz_caasced.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

     for (x,y,w,h) in yüz_rect:
          cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),10)
     cv2.imshow("yüz tespiti",frame)

     if cv2.waitKey(1) & 0XFF==ord("q"):
          break

video.release()
cv2.destroyAllWindows()

plt.show()
