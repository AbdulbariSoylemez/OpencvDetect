import cv2
import os

img_folder = "YayaTespiti/YayaImg"  # Resimlerin bulunduğu klasörün yolu

yayapath = []

for dosya in os.listdir(img_folder):
    if dosya.endswith(".jpg"):
        yayapath.append(os.path.join(img_folder, dosya))

print(yayapath)

# hog tanımlayıcısı
hog = cv2.HOGDescriptor()
# tanımlayıcı
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

for imgpath in yayapath:
    image = cv2.imread(imgpath)

    if image is not None:
        (rects, weights) = hog.detectMultiScale(image, padding=(8, 8), scale=1.025)

        for (x, y, w, h) in rects:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)

        cv2.imshow("yaya", image)
        
        key = cv2.waitKey(0)
        if key & 0xFF == ord("q"):continue 
            
    else:
        print(f"Hata: {imgpath} dosyası yüklenemedi.")

cv2.destroyAllWindows()
