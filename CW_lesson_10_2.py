import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


colors = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'black': (0, 0, 0)
}


X = []
y = []
for color_name,bgr in colors.items():
    for _ in range(20):
        noise = np.random.randint(-20, 20, 3)
        sample = np.clip(np.array(bgr)+noise, 0,255)
        X.append(sample)
        y.append(color_name)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 2)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (20,50,50), (255,255,255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))

    countours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for nct in countours:
        area = cv2.contourArea(nct)
        if area > 1200:
            x, y, w, h = cv2.boundingRect(nct)
            roi = frame[y:y+h, x:x+w]
            mean_color = cv2.mean(roi)[:3]
            mean_color = np.array(mean_color).reshape(1,-1)

            label = model.predict(mean_color)[0]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
            cv2.putText(frame, label.upper(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)


    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
