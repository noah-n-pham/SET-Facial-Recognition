import os
import cv2
import sys
from zipfile import ZipFile
from urllib.request import urlretrieve
import time
import math


# ========================-Downloading Assets-========================
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


def rotated_dimensions(width, height, angle_degrees):
    angle_rad = math.radians(angle_degrees)
    new_width = abs(width * math.cos(angle_rad)) + \
        abs(height * math.sin(angle_rad))
    new_height = abs(width * math.sin(angle_rad)) + \
        abs(height * math.cos(angle_rad))
    return int(new_width), int(new_height)


def rotate_image(img, angle):
    percentage = 0.7
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nW = int(((h * sin) + (w * cos)) * percentage)
    nH = int(((h * cos) + (w * sin)) * percentage)
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    return cv2.warpAffine(img, M, (nW, nH))


# Download if assest ZIP does not exists.
if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)
# ====================================================================


s = 1
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
    score_threshold=0.4,
    nms_threshold=0.3,
    top_k=5000
)
centerCoordinates = (0, 0)
dimentions = [0, 0]

width = 224
height = 224


def face_detection(source, image_name=None):
    image = None

    has_frame, frame = source.read()
    if not has_frame:
        return
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
        
        # Add offset (10% padding)
        offset = 10
        x_offset = int((offset/100) * w)
        y_offset = int((offset/100) * h)
        
        x_padded = x - x_offset
        y_padded = y - y_offset
        w_padded = w + 2 * x_offset
        h_padded = h + 2 * y_offset
        
        # Ensure bounds are within frame test
        x_padded = max(0, x_padded)
        y_padded = max(0, y_padded)
        w_padded = min(w_padded, frame_width - x_padded)
        h_padded = min(h_padded, frame_height - y_padded)
        
        if image_name is not None:
            image = frame[y_padded:y_padded+h_padded, x_padded:x_padded+w_padded]
            image = cv2.resize(image, (width, height))
            cv2.imwrite(image_name, image)
            dimentions[0] = w_padded
            dimentions[1] = h_padded
        
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
    
    cv2.imshow(win_name, frame)
    # centerCoordinates = (nose_x, nose_y)
    return image


def get_coordinates():
    return centerCoordinates


def image_dimentions():
    return dimentions


i = 0

nameDict = {}
name = input("Enter file name: ")
nameDict[name] = i
folderName = "data/raw/Dataset/hoek"
while True:
    face_detection(source)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break
    elif key == 13:
        name = input("Enter file name: ")
        i = nameDict[name] if name in nameDict else 0
    elif key == 32:
        # face_detection(source, folderName + "/"+name+"_" + str(i) + ".png")
        # image = cv2.imread(folderName + "/"+name+"_" + str(i) + ".png")
        # print(image.shape)
        # # image = cv2.resize(image, (300, 300))
        # cv2.imwrite(folderName + "/"+name+"_" + str(i + 1) +
        #             ".png", rotate_image(image, 22))
        # cv2.imwrite(folderName + "/"+name+"_" + str(i + 2) + ".png",
        #             cv2.convertScaleAbs(image, alpha=1.5, beta=0))
        # cv2.imwrite(folderName + "/"+name+"_" + str(i + 3) + ".png",
        #             cv2.convertScaleAbs(image, alpha=0.6, beta=0))
        # cv2.imwrite(folderName + "/"+name+"_" + str(i + 4) + ".png",
        #             cv2.GaussianBlur(image, (11, 11), 0))
        # i += 1
        nameDict[name] = i

        # cv2.imwrite()

source.release()
cv2.destroyWindow(win_name)
