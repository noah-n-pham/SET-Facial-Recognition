from ultralytics import YOLO
import cv2
import serial

#port = '/dev/ttyACM0' 

#arduino = serial.Serial(port=port, baudrate=9600, timeout=1)


# Load YOLOv8n model
model = YOLO("yolov8n.pt")

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()
'''
while True:
        ret, frame = cap.read() # Read a frame

        if not ret: # If frame reading fails, break the loop
            print("Error: Could not read frame.")
            break

        cv2.imshow('Camera Feed', frame) # Display the frame

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
'''

while True:
    success, frame = cap.read()
    if not success:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame, stream=True)

    # Loop through results and draw boxes
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extract coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Extract confidence and class
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        annotated_frame = r.plot()  # returns frame with boxes/labels drawn

        # Display the annotated frame
        cv2.imshow("YOLOv8 Real-Time", annotated_frame)

        #time.sleep(2)
        # Send the result to Arduino
        x,y = (x1 + x2)/2, (y1 + y2)/2
        message = f"{x},{y}"
        #arduino.write((message + '\n').encode())
        print("Sent:", message)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
