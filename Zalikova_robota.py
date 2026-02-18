import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

zona = np.array([(0, 0),(1500, 0),(1500, 1000),(0, 1000)], np.int32)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4, verbose=False)

    cv2.polylines(frame, [zona], True, (0, 0, 255), 2)
    cv2.putText(frame, "FORBIDDEN ZONE", (200, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])

            if cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Центр людини
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                inside = cv2.pointPolygonTest(zona, (cx, cy), False)

                if inside >= 0:
                    color = (0, 0, 255)
                    label = "WARNING!"
                else:
                    color = (0, 255, 0)
                    label = "Person"

                # Малюємо bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 5, color, -1)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("YOLO", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
