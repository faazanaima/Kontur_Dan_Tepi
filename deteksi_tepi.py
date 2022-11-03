# FAAZA NAIMA 41520120010
# Deteksi Tepi Geometri Sederhana
from matplotlib import pyplot as plt
import argparse
import imutils
import cv2

# Argumen
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help= "path to the input image")

args = vars(ap.parse_args())

# load image, konversi ke grayscale -> blur -> threshold untuk mendapatkan citra biner

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
thres = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

# Proses menemukan kontur di threshold image
cnts = cv2.findContours(thres.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnts = imutils.grab_contours(cnts)

# Proses pengulangan di variabel kontur (cnts)

for c in cnts:
    if cv2.contourArea(c)>2:
        M = cv2.moments(c)
        cX = int(M["m10"]/M["m00"])
        cY = int(M["m01"]/M["m00"])

# Menggambar kontur dan titik tengah dari bentuk pada gambar
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.circle(image, (cX, cY), 7, (255,255,255), -1)
        cv2.putText(image, "center", (cX - 20, cY - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)

plt.subplot(1,2,1), plt.imshow(thres, cmap='gray')
plt.title('Binary Image'), plt.xticks([]), plt.yticks([])

plt.subplot(1,2,2), plt.imshow(image, cmap='gray')
plt.title('Gambar'), plt.xticks([]), plt.yticks([])

plt.show()

# cv2.imshow("Binary Image", thres)
# cv2.imshow("Gambar", image)
# cv2.waitKey(0)
# cv2.destroyaAllWindows()