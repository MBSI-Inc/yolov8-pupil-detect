import cv2
import math
from ultralytics import YOLO
from sys import platform

def predict_webcam():
    print("Start predict with webcam")
    # For convenience, you can copy the best.pt file to the root folder and rename
    # it to have easier access
    model = YOLO("yolov8_pupil_weight.pt")
    # The next section is setup the camera
    if platform == "win32":
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    cam.set(cv2.CAP_PROP_FPS, 15)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    class_name = ['Iris', 'R Eye', 'L Eye', 'Pupil']
    class_color = [(255, 0, 0), (0, 255, 0), (0, 255, 0), (0, 0, 255)]

    while True:
        _, img = cam.read()
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
                confidence = math.ceil((box.conf[0]*100))/100
                class_index = int(box.cls[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), class_color[class_index], 2)
                print("Class name: ", class_name[class_index], "| Confidence: ",confidence)
                cv2.putText(img, class_name[class_index], [x1, y1], cv2.FONT_HERSHEY_SIMPLEX, 1, class_color[class_index], 2)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    predict_webcam()