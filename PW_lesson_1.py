import cv2
import numpy as np

image = np.full((500, 700, 3),(204,204,204), np.uint8)
cv2.rectangle(image, (15, 15), (685, 485), (76,76,76), 3)
photo = cv2.imread("image_3.png")
x, y, w, h = 40, 60, 160, 180
resized = cv2.resize(photo, (w, h))
image[y:y+h, x:x+w] = resized
cv2.putText(image, "Yaremenko Oleksandr", (240, 115), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
cv2.putText(image, "Computer Vision Student", (240, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
cv2.putText(image, "Email: oleksandr.yaremenko@gmail.com", (240, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (75, 75, 75), 1)
cv2.putText(image, "Phone: +380980936131", (240, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (75, 75, 75), 1)
cv2.putText(image, "30.06.2010", (239, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (75, 75, 75), 1)
cv2.putText(image, "OpenCV Business Card", (230, 470), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (75, 75, 75), 2)
qr = cv2.imread("qr.png")
qr = cv2.resize(qr, (100, 100))
image[300:400, 550:650] = qr

cv2.imwrite("business_card.png", image)
cv2.imshow("Business Card", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
