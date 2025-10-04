import cv2
image = cv2.imread('images/mister.png')
size = cv2.resize(image, (600, 1000))
cv2.imshow('image', size)
cv2.imshow("window",image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)
border = cv2.Canny(size, 50, 150, apertureSize = 3)
cv2.imshow("border", border)
rotate = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
cv2.imshow("rotate", rotate)
blur = cv2.blur(gray, (7, 7), 0)
cv2.imshow("blur", blur)


cv2.waitKey(0)
cv2.destroyAllWindows()