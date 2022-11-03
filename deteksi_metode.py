# FAAZA NAIMA 41520120010
# DETEKSI TEPI PADA GAMBAR DENGAN METODE SOBEL, LAPLACIAN, CANNY

# Multidimensional image processing : Sobel, prewitt, 
# from scipy import ndimage 
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np

# Mengubah gambar menjadi grayscale dan blur untuk menghilangkan noise
img = cv.imread("lena.png", cv.IMREAD_GRAYSCALE)
img = cv.GaussianBlur(img, (11, 11), 0)

# # metode sobel
# sobelx = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
# sobely = np.array([[1,2,1], [0,0,0], [-1,0,1]])
# img_sobelx = cv.filter2D(img, -1, sobelx)
# img_sobely = cv.filter2D(img, -1, sobely)
# sobel = cv.add(img_sobelx, img_sobely)

# metode sobel dengan library openCV
sobelx = cv.Sobel(img, cv.CV_64F, 1, 0)
sobely = cv.Sobel(img, cv.CV_64F, 0, 1)
sobel = cv.add(sobelx, sobely)

# metode prewitt
prewitx = np.array([[-1,0,1], [-1,0,1], [-1,0,1]])
prewity = np.array([[1,1,1], [0,0,0], [-1,-1,-1]])
img_prewitx = cv.filter2D(img, -1, prewitx)
img_prewity = cv.filter2D(img, -1, prewity)
prewit = cv.add(img_prewitx, img_prewity)

# metode robert
robertx = np.array([[1,0], [0,-1]])
roberty = np.array([[0,1], [-1,0]])
img_robertx = cv.filter2D(img, -1, robertx)
img_roberty = cv.filter2D(img, -1, roberty)
robert = cv.add(img_robertx, img_roberty)

# metode laplacian dengan library openCV
laplacian = cv.Laplacian(img, cv.CV_64F, ksize=5)

# metode canny
canny = cv.Canny(img, 100, 150) 

# menampilkan gambar
plt.subplot(2,3,1), plt.imshow(img, cmap='gray')
plt.title('Gambar'), plt.xticks([]), plt.yticks([])

plt.subplot(2,3,2), plt.imshow(sobel, cmap='gray')
plt.title('Sobel'), plt.xticks([]), plt.yticks([])

plt.subplot(2,3,3), plt.imshow(prewit, cmap='gray')
plt.title('Prewitt'), plt.xticks([]), plt.yticks([])

plt.subplot(2,3,4), plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

plt.subplot(2,3,5), plt.imshow(robert, cmap='gray')
plt.title('Robert'), plt.xticks([]), plt.yticks([])

plt.subplot(2,3,6), plt.imshow(canny, cmap='gray')
plt.title('Canny'), plt.xticks([]), plt.yticks([]) 
 
plt.show()