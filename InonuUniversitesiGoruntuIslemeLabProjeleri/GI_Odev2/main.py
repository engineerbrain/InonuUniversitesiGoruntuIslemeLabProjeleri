import cv2
import numpy as np

# Kamera girişi
cap = cv2.VideoCapture(0)

while True:
    # Görüntüyü oku
    ret, frame = cap.read()

    # Görüntüyü HSV renk uzayına dönüştür
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Kırmızı rengin HSV renk aralığını belirle
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # Renk aralığına uygun maske oluştur
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Bitwise AND işlemi ile orijinal görüntüde belirtilen renk aralığına uygun pikselleri göster
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Görüntüleri ekranda göster
    cv2.imshow('Original', frame)
    cv2.imshow('Result', result)

    # 'q' tuşuna basıldığında çıkış yap
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
