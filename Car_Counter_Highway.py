from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import numpy as np

carvideo = cv2.VideoCapture("highway.mov")
model = YOLO("Yolo-Weights/yolov8m.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
              "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
              "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
              "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
              "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
              "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
              "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
              "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
              "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

mask = cv2.imread("mask-highway.jpg")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
lineLimitsDown = [600, 1700, 1600, 1700]
lineLimitsUp = [1750, 1600, 2700, 1600]
totalCountDown = []
totalCountUp = []

# Trajectory storage
trajectories = {}

# Define the codec and create a VideoWriter object
output_path = "output_video_first.avi"
fourcc = cv2.VideoWriter_fourcc(*"XVID")
output_video = cv2.VideoWriter(output_path, fourcc, 30.0, (int(carvideo.get(3)), int(carvideo.get(4))))

while True:
    success, img = carvideo.read()
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    imgRegion = cv2.bitwise_and(img, mask)
    imgGraphics = cv2.imread("car_graphic1.png", cv2.IMREAD_UNCHANGED)
    imgGraphics2 = cv2.imread("car_graphic2.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    img = cvzone.overlayPNG(img, imgGraphics2, (3272, 0))
    results = model(imgRegion, device="mps")

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass in ["car", "motorbike", "truck"] and conf > 0.4:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    cv2.line(img, (lineLimitsDown[0], lineLimitsDown[1]), (lineLimitsDown[2], lineLimitsDown[3]), (0, 0, 255), 25)
    cv2.line(img, (lineLimitsUp[0], lineLimitsUp[1]), (lineLimitsUp[2], lineLimitsUp[3]), (0, 0, 255), 25)

    for result in resultsTracker:
        x1, y1, x2, y2, track_id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        # Draw smaller and more frequent dashed trajectory
        if track_id in trajectories:
            for i in range(1, len(trajectories[track_id])):
                if i % 2 == 0:  # Draw dashed line every 2 iterations
                    cv2.line(img, trajectories[track_id][i - 1], trajectories[track_id][i], (0, 255, 0), 2)
            trajectories[track_id].append((x1 + w // 2, y1 + h // 2))

        else:
            trajectories[track_id] = [(x1 + w // 2, y1 + h // 2)]

        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 5)
        
        # Create the text
        text = f'ID:{int(track_id)}'

        # Get the size of the text
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, fontScale=2, thickness=1)[0]

        # Set the size of the rectangle
        rect_size = (text_size[0] + 10, text_size[1] + 10)

        # Calculate the corners of the rectangle
        rect_start = (x1-2, y1 - text_size[1] - 5)
        rect_end = (x1-2 + rect_size[0], y1)

        # Draw the rectangle
        cv2.rectangle(img, rect_start, rect_end, (0, 255, 0), thickness=cv2.FILLED)

        # Add the text
        cv2.putText(img, text, (x1, y1), cv2.FONT_HERSHEY_PLAIN, fontScale=2, thickness=1, color=(255, 0, 0))

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if lineLimitsDown[0] < cx < lineLimitsDown[2] and lineLimitsDown[1] - 30 < cy < lineLimitsDown[1] + 30:
            if track_id not in totalCountDown:
                totalCountDown.append(track_id)
                cv2.line(img, (lineLimitsDown[0], lineLimitsDown[1]), (lineLimitsDown[2], lineLimitsDown[3]), (0, 255, 0), 25)

        if lineLimitsUp[0] < cx < lineLimitsUp[2] and lineLimitsUp[1] - 30 < cy < lineLimitsUp[1] + 30:
            if track_id not in totalCountUp:
                totalCountUp.append(track_id)
                cv2.line(img, (lineLimitsUp[0], lineLimitsUp[1]), (lineLimitsUp[2], lineLimitsUp[3]), (0, 255, 0), 25)

    if len(totalCountDown) < 10:
        cv2.putText(img, str(len(totalCountDown)), (255, 155), cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 0), 12)
    else:
        cv2.putText(img, str(len(totalCountDown)), (235, 140), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 0), 8)
    
    if len(totalCountUp) < 10:
        cv2.putText(img, str(len(totalCountUp)), (3485, 155), cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 0), 12)
    else:
        cv2.putText(img, str(len(totalCountUp)), (3440, 140), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 0), 8)
    cv2.imshow("Video", img)
    output_video.write(img)  # Write frame to the output video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoWriter and close all windows
output_video.release()
carvideo.release()
cv2.destroyAllWindows()
