import cv2
import matplotlib.pyplot as plt
import numpy as np

import os

img_folder = "catDetect/kediImg"  # Resimlerin bulunduğu klasörün yolu

kedipath = []

for dosya in os.listdir(img_folder):
    if dosya.endswith(".jpg"):
        kedipath.append(os.path.join(img_folder, dosya))

print(kedipath)

for j in kedipath:
    print(j)
    img = cv2.imread(j)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)
    dedector=cv2.CascadeClassifier("haarcascade_frontalcatface.xml")
    yuzcased=dedector.detectMultiScale(gray,scaleFactor=1.0245, minNeighbors=1)

    for (i,(x,y,w,h)) in enumerate(yuzcased):# enumerate sayesinde index değerinide alabiliriz 
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),5)
        cv2.putText(img,"kedi {}".format(i+1),(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,255),2)
        


    cv2.imshow(dosya, img)
    if cv2.waitKey(0) & 0xFF == ord("q"):continue
    
    
cv2.destroyAllWindows()
