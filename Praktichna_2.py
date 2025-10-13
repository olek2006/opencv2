import cv2
import numpy as np


img = cv2.imread('image/objects.jpg')
img_copy = img.copy()
img = cv2.GaussianBlur(img_copy, (5, 5), 2)
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_black = np.array([0,0,0])
upper_black = np.array([179,255,80])

lower_blue = np.array([68,42,90])
upper_blue = np.array([139,255,255])

lower_red = np.array([166,0,0])
upper_red = np.array([179,255,255])

lower_green = np.array([0,0,0])
upper_green = np.array([56,0,0])

mask_black = cv2.inRange(img, lower_black, upper_black)
mask_blue = cv2.inRange(img, lower_blue, upper_blue)
mask_green = cv2.inRange(img, lower_green, upper_green)
mask_red = cv2.inRange(img, lower_red, upper_red)
mask_total = cv2.bitwise_or(mask_red, mask_blue)
mask_total = cv2.bitwise_or(mask_total, mask_green)
mask_total = cv2.bitwise_or(mask_total, mask_black)


green_countours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in green_countours:
    area = cv2.contourArea(cnt)
    if area > 100:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.putText(img_copy, "Color: black", (x, y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

black_countours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in black_countours:
    area = cv2.contourArea(cnt)
    if area > 100:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.putText(img_copy, "Color: black", (x, y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

red_contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in red_contours:
    area = cv2.contourArea(cnt)
    if area > 100:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.putText(img_copy, "Color: red", (x, y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

blue_contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in blue_contours:
    area = cv2.contourArea(cnt)
    if area > 100:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.putText(img_copy, "Color: blue", (x, y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 200:
        x, y, w, h = cv2.boundingRect(cnt)

        perimetr = cv2.arcLength(cnt, True)
        M = cv2.moments(cnt)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])



        aspect_ratio = round(w / h, 2)
        compactness = round((4 * np.pi * area) / (perimetr ** 2), 2)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimetr, True)
        if len(approx) == 4:
            shape = "squad"
        elif len(approx) == 3:
            shape = "triangle"
        elif len(approx) >= 8:
            shape = "oval"
        else:
            shape = "other"


        cv2.putText(img_copy,f'S:{int(area)}, P{int(perimetr)}', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2 )
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(img_copy, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(img_copy, f'AR:{aspect_ratio}, C:{compactness}', (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2 )
        cv2.putText(img_copy, f'shape: {shape}', (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2 )

cv2.imshow('mask', img_copy)

cv2.imshow('img', img)
cv2.imwrite('result.jpg', img_copy)

cv2.waitKey(0)

cv2.destroyAllWindows()
