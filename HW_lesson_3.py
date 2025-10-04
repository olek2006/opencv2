import cv2
image = cv2.imread('images/image_3.png')
image = cv2.rectangle(image, (190, 200), (410, 495), (14,14,237), 4)
image = cv2.putText(image, "Yaremenko Oleksandr", (165,520), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (14,14,237), 2)
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()