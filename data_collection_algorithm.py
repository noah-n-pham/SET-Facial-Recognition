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
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    return cv2.warpAffine(img, M, (nW, nH))


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

net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")
# Model parameters
centerCoordinates = (0, 0)
dimentions = [0, 0]


def face_detection(source, image_name=None):
    in_width = 300
    in_height = 300
    mean = [104, 117, 123]
    conf_threshold = 0.4

    has_frame, frame = source.read()
    if not has_frame:
        return
    frame = cv2.flip(frame, 1)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (in_width, in_height), mean, swapRB=False, crop=False)
    # Run a model
    net.setInput(blob)
    detections = net.forward()

    x_top_left = 0
    y_top_left = 0
    x_bottom_right = 0
    y_bottom_right = 0

    confidence = detections[0, 0, 0, 2]
    if confidence > conf_threshold:
        x_top_left = int(detections[0, 0, 0, 3] * frame_width)
        y_top_left = int(detections[0, 0, 0, 4] * frame_height)
        x_bottom_right = int(detections[0, 0, 0, 5] * frame_width)
        y_bottom_right = int(detections[0, 0, 0, 6] * frame_height)

        image = None
        if (image_name != None):
            image = frame[y_top_left:y_bottom_right, x_top_left:x_bottom_right]
            cv2.imwrite(image_name, image)
            dimentions[0] = x_bottom_right - x_top_left
            dimentions[1] = y_bottom_right - y_top_left
        image_copy = image

        cv2.rectangle(frame, (x_top_left, y_top_left),
                      (x_bottom_right, y_bottom_right), (0, 255, 0))
        label = "Confidence: %.4f" % confidence
        label_size, base_line = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        cv2.rectangle(
            frame,
            (x_top_left, y_top_left - label_size[1]),
            (x_top_left + label_size[0], y_top_left + base_line),
            (255, 255, 255),
            cv2.FILLED,
        )
        cv2.putText(frame, label, (x_top_left, y_top_left),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    t, _ = net.getPerfProfile()
    cv2.circle(frame, (int((x_top_left + x_bottom_right)/2),
               int((y_top_left + y_bottom_right)/2)), 5, (255, 0, 0), -1)
    cv2.imshow(win_name, frame)
    # centerCoordinates = (int((x_top_left + x_bottom_right)/2), int((y_top_left + y_bottom_right)/2))
    return image_copy


def get_coordinates():
    return centerCoordinates


def image_dimentions():
    return dimentions


i = 0

nameDict = {}
name = input("Enter file name: ")
nameDict[name] = i

while True:
    face_detection(source)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break
    elif key == 13:
        name = input("Enter file name: ")
        i = nameDict[name] if name in nameDict else 0
    elif key == 32:
        face_detection(source, "test/"+name+"_" + str(i) + ".png")
        image = cv2.imread("test/"+name+"_" + str(i) + ".png")
        cv2.imwrite("test/"+name+"_" + str(i + 1) +
                    ".png", rotate_image(image, 22))
        cv2.imwrite("test/"+name+"_" + str(i + 2) + ".png",
                    cv2.convertScaleAbs(image, alpha=1.5, beta=0))
        cv2.imwrite("test/"+name+"_" + str(i + 3) + ".png",
                    cv2.convertScaleAbs(image, alpha=0.6, beta=0))
        cv2.imwrite("test/"+name+"_" + str(i + 4) + ".png",
                    cv2.GaussianBlur(image, (11, 11), 0))
        i += 6
        nameDict[name] = i

        # cv2.imwrite()

source.release()
cv2.destroyWindow(win_name)
