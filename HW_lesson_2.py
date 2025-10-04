import cv2


image = cv2.imread('images/image_2.png')
print(image.shape)

image = cv2.resize(image, (400,400))

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.Canny(image, 40, 40)
image = cv2.imshow('Oleksandr', image)

image2 = cv2.imread('images/image_2_1.png')
print(image2.shape)
image2 = cv2.resize(image2, (500,200))
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
image2 = cv2.Canny(image2, 125, 125)

image2 = cv2.imshow('Email', image2)

cv2.waitKey(0)
cv2.destroyAllWindows()