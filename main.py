import cv2
import numpy as np

cap = cv2.VideoCapture(0)

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if ret is False:
        break

    frame = cv2.flip(frame, 1)  # Remove if the camera is mirrored

    rows, cols, _ = frame.shape
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (7, 7), 0)

    _, threshold = cv2.threshold(gray_frame, 1, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)

        cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 3)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.line(frame, (x + int(w/2), 0), (x + int(w/2), rows), (255, 255, 0), 2)
        cv2.line(frame, (0, y + int(h/2)), (cols, y + int(h/2)), (255, 255, 0), 2)

        center_coordinates = (x + w // 2, y + h // 2)
        radius = 25  # Radius of the circle
        color = (0, 255, 0)
        thickness = 1

        cv2.circle(frame, center_coordinates, radius, color, thickness)

        label = f"Eye: x={x}, y={y}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        break

    # cv2.imshow("Threshold", threshold)  # Uncomment if you want to see threshold debug
    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
