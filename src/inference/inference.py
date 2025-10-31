import os
import cv2
import sys
from zipfile import ZipFile
from urllib.request import urlretrieve
import serial
import time

#Configure the serial port
#Replace 'COMx' with your Arduino's serial port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux/Mac)
#Ensure the baud rate matches the Arduino sketch
#ser = serial.Serial('COM3', 9600)
#time.sleep(2)  # Give Arduino time to reset after opening serial


#def send_coordinates(x, y):
    # Format the coordinates as a string, e.g., "X100Y200\n"
    # The newline character '\n' helps the Arduino detect the end of a message
    #message = f"X{x}Y{y}\n"
    #ser.write(message.encode())  # Encode the string to bytes before sending
    #print(f"Sent: {message.strip()}")



# ========================-Downloading Assets-========================
def get_coordinates():
    return centerCoordinates
def download_and_unzip(url, save_path):
    print(f"Downloading and extracting assests....", end="")


    # Downloading zip file using urllib package.
    urlretrieve(url, save_path)


    try:
        # Extracting zip file using the zipfile package.
        with ZipFile(save_path) as z:
            # Extract ZIP file contents in the same directory.
            z.extractall(os.path.split(save_path)[0])


        print("Done")


    except Exception as e:
        print("\nInvalid file.", e)




URL = r"https://www.dropbox.com/s/efitgt363ada95a/opencv_bootcamp_assets_12.zip?dl=1"


asset_zip_path = "assets/opencv_bootcamp_assets_12.zip"


# Download if assest ZIP does not exists.
if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)
# ====================================================================




s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]


source = cv2.VideoCapture(s)


win_name = "Camera Preview"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)


# Initialize YuNet detector
detector = cv2.FaceDetectorYN.create(
    model="assets/face_detection_yunet_2023mar.onnx",
    config="",
    input_size=(320, 320),
    score_threshold=0.7,
    nms_threshold=0.3,
    top_k=5000
)
centerCoordinates = (0, 0)

while cv2.waitKey(1) != 27:
    has_frame, frame = source.read()
    if not has_frame:
        break
    frame = cv2.flip(frame, 1)
    frame_height, frame_width = frame.shape[:2]
    
    # Update input size for current frame
    detector.setInputSize((frame_width, frame_height))
    
    # Detect faces
    _, faces = detector.detect(frame)
    
    if faces is not None and len(faces) > 0:
        # Get the most confident face (first one after NMS)
        face = faces[0]
        
        # Extract bounding box and confidence
        x, y, w, h = map(int, face[:4])
        confidence = face[14]  # YuNet stores normalized confidence at index 14
        
        # Extract landmarks: right_eye, left_eye, nose, right_mouth, left_mouth
        nose_x = int(face[8])   # Nose tip x-coordinate
        nose_y = int(face[9])   # Nose tip y-coordinate
        
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw confidence label
        label = "Confidence: %.4f" % confidence
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            frame,
            (x, y - label_size[1]),
            (x + label_size[0], y + base_line),
            (255, 255, 255),
            cv2.FILLED,
        )
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        
        # Draw center point at nose landmark (more accurate than bbox center)
        cv2.circle(frame, (nose_x, nose_y), 5, (255, 0, 0), -1)
        
        # Update center coordinates
        centerCoordinates = (nose_x, nose_y)
        fps_label = "Center Coordinates: x=%d y=%d" % (nose_x, nose_y)
        cv2.putText(frame, fps_label, (0, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        
        #send_coordinates(centerCoordinates[0], centerCoordinates[1])
    
    cv2.imshow(win_name, frame)


source.release()
cv2.destroyWindow(win_name)

