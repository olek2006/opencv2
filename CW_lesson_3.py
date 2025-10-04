import cv2
import numpy as np

image = np.zeros((500, 400, 3), np.uint8)
cv2.imshow("image", image)
#image[:] = (85,214,92)
#rgb = bgr
# image[100:150, 200:250] = (85,214,92)
cv2.rectangle(image, (100, 100), (200, 200), (85,214,92), 1)

cv2.line(image, (100, 100), (200, 200), (85,214,92), 1)

print(image.shape)
cv2.line(image, (0, image.shape[0] // 2), (image.shape[1], image.shape[0] // 2), (85,214,92), 1)

cv2.line(image, (image.shape[1] // 2, 0), (image.shape[1] // 2, image.shape[0]), (85,214,92), 1)

cv2.circle(image, (200, 200), 30, (85,214,92), 2)

cv2.putText(image, "2144267346", (10, 150), cv2.FONT_HERSHEY_PLAIN, 2, (85,214,92), 2)
cv2.putText(image, "Oleksandr Yaremenko", (30, 250), cv2.FONT_HERSHEY_PLAIN, 2, (85,214,92), 2)
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()