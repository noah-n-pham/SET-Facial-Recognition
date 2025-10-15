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


asset_zip_path = os.path.join(os.getcwd(), f"opencv_bootcamp_assets_12.zip")


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


net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")
# Model parameters
in_width = 300
in_height = 300
mean = [104, 117, 123]
conf_threshold = 0.7
centerCoordinates = (0, 0)

while cv2.waitKey(1) != 27:
    has_frame, frame = source.read()
    if not has_frame:
        break
    frame = cv2.flip(frame, 1)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]


    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB=False, crop=False)
    # Run a model
    net.setInput(blob)
    detections = net.forward()

    confidence = detections[0, 0, 0, 2]
    if confidence > conf_threshold:
        x_top_left = int(detections[0, 0,0, 3] * frame_width)
        y_top_left = int(detections[0, 0, 0, 4] * frame_height)
        x_bottom_right  = int(detections[0, 0, 0, 5] * frame_width)
        y_bottom_right  = int(detections[0, 0, 0, 6] * frame_height)
        cv2.rectangle(frame, (x_top_left, y_top_left), (x_bottom_right, y_bottom_right), (0, 255, 0))
        label = "Confidence: %.4f" % confidence
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)


        cv2.rectangle(
                frame,
                (x_top_left, y_top_left - label_size[1]),
                (x_top_left + label_size[0], y_top_left + base_line),
                (255, 255, 255),
                cv2.FILLED,
            )
        cv2.putText(frame, label, (x_top_left, y_top_left), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))


    t, _ = net.getPerfProfile()
    fps_label = "Center Coordinates: x=%d y=%d" % (int(x_top_left/2), int(y_top_left/2))
    cv2.putText(frame, fps_label, (0, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    cv2.imshow(win_name, frame)
    cv2.circle(frame, (int((x_top_left + x_bottom_right)/2), int((y_top_left + y_bottom_right)/2)), 5, (255, 0, 0), -1)
    cv2.imshow(win_name, frame)
    
    centerCoordinates = (int((x_top_left + x_bottom_right)/2), int((y_top_left + y_bottom_right)/2))
    #send_coordinates(centerCoordinates[0], centerCoordinates[1])


source.release()
cv2.destroyWindow(win_name)

