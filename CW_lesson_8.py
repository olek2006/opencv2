import cv2
import numpy as np

# face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
# face_cascade = cv2.CascadeClassifier('data/lbpcascade_frontalface_improved.xml')
# eye_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')
# smile_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_smile.xml')
face_net = cv2.dnn.readNetFromCaffe("data/DNN/deploy.prototxt", "data/DNN/res10_300x300_ssd_iter_140000.caffemodel")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # #scalefactor - коефіцієнт масштабування minNeighbors - кількість перевірок
    # faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5, minSize = (30, 30))
    # # print(faces)
    # for (x,y,w,h) in faces:
    #     cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    #     roi_gray = gray[y:y+h, x:x+w]
    #     roi_color = frame[y:y+h, x:x+w]
    #     eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10, minSize = (15,15))
    #     for (ex,ey,ew,eh) in eyes:
    #         cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
    #     smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 10, minSize = (25,25))
    #     for (sx,sy,sw,sh) in smiles:
    #         cv2.rectangle(roi_color, (sx,sy), (sx+sw,sy+sh), (0,0,0), 2)
    #     cv2.putText(frame, f"Faces detected: {len(faces)}", (x,y), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255))

    if not ret:
        break
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)

    detections = face_net.forward()
    # print(detections)
    for i in range(detections.shape[2]):


    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()