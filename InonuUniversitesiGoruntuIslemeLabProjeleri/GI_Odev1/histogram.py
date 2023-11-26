import cv2
import numpy as np
import matplotlib.pyplot as plt


resim = cv2.imread("images.jpg",0)
cv2.imshow("resim",resim)
cv2.waitKey()
def histogram_hesapla(goruntu):
    w, h = goruntu.shape
    histogram = np.zeros(256)

    for u in range(w):
        for v in range(h):
            piksel_degeri = goruntu[u, v]
            histogram[piksel_degeri] += 1

    return histogram

def kumulatif_histogram_hesapla(goruntu):
    w, h = goruntu.shape
    histogram = np.zeros(256)
    kumulatif_histogram = np.zeros(256)

    for u in range(w):
        for v in range(h):
            piksel_degeri = goruntu[u, v]
            histogram[piksel_degeri] += 1

    kumulatif_sum = 0
    for i in range(256):
        kumulatif_sum += histogram[i]
        kumulatif_histogram[i] = kumulatif_sum

    normal_histogram = histogram_hesapla(goruntu)

    return kumulatif_histogram, normal_histogram


kumulatif_histogram, normal_histogram = kumulatif_histogram_hesapla(resim)


plt.plot(range(256), kumulatif_histogram)
plt.title("Kümülatif histogram")
plt.show()

plt.bar(range(256), normal_histogram)
plt.title("Normal histogram")
plt.show()
