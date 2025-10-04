import cv2
import numpy as np

image = np.full((400, 600, 3),(204,204,204), np.uint8)
cv2.rectangle(image, (10, 10), (590, 390), (76,76,76), 3)
photo = cv2.imread("images/image_3.png")
x, y, w, h = 40, 60, 130, 150
resized = cv2.resize(photo, (w, h))
image[y:y+h, x:x+w] = resized
cv2.putText(image, "Yaremenko Oleksandr", (205, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
cv2.putText(image, "Computer Vision Student", (205, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
cv2.putText(image, "Email: oleksandr.yaremenko@gmail.com", (205, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (75, 75, 75), 1)
cv2.putText(image, "Phone: +380980936131", (205, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (75, 75, 75), 1)
cv2.putText(image, "30.06.2010", (205, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (75, 75, 75), 1)
cv2.putText(image, "OpenCV Business Card", (205, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (75, 75, 75), 2)
qr = cv2.imread("qr.png")
qr = cv2.resize(qr, (75, 75))
image[240:315, 460:535] = qr

cv2.imwrite("business_card.png", image)
cv2.imshow("Business Card", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
