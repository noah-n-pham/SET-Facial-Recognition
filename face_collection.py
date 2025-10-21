import cv2
import sys

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    cv2.namedWindow("PAPPLE", cv2.WINDOW_NORMAL)

    cv2.imshow("YuNet Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()