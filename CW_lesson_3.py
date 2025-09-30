import cv2
import numpy as np
img = np.zeros((500,400,3), np.uint8)
# img[:] = 101,243,43
# img[100:150, 200:250] = 101,243,43


cv2.rectangle(img,(200,200),(390,300),(101,243,43),1)
cv2.line(img,(200,200),(390,300),(101,243,43),1)
# cv2.line(img,(390,200),(200,300),(101,243,43),1)
print(img.shape)
cv2.line(img,(0,img.shape[0]//2),(img.shape[1],img.shape[0]//2),(101,243,43),1)
cv2.line(img,(img.shape[1]//2,0),(img.shape[1]//2,img.shape[0]),(101,243,43),1)
cv2.circle(img,(200,250), 30, (101,243,43),-1)
cv2.putText(img, "Yaremenko Oleksandr", (200,150), cv2.FONT_HERSHEY_PLAIN, 1, (101,243,43))
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()