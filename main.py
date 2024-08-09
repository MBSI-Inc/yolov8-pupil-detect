import cv2
import math
from ultralytics import YOLO
from sys import platform

def train():
    model = YOLO("yolov8m.pt")
    model.train(data="datasets/data.yaml", epochs=20)
    model.val()

def predict():
    print("Start predict")
    model = YOLO("runs/detect/train9/weights/best.pt")
    # Run batched inference on a list of images
    results = model(["test_stuff/im2.jpg"])  # return a list of Results objects

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen
        result.save(filename="test_stuff/result.png")  # save to disk


def predict_webcam():
    print("Start predict with webcam")
    # model = YOLO("runs/detect/train9/weights/best.pt")
    model = YOLO("yolov8_pupil_weight.pt")
    if platform == "win32":
        cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    else:
        cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    cam.set(cv2.CAP_PROP_FPS, 15)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    # The cau001 dataset can detect L and R eye separately
    class_name = ['Iris', 'R Eye', 'L Eye', 'Pupil'] 
    class_color = [(255, 0, 0), (0, 255, 0), (0, 255, 0), (0, 0, 255)]

    box_len = 0

    while True:
        _, img = cam.read()
        results = model(img, stream=True)
        for r in results: # results is a Generator Function
            best_iris_data = [0, 0, 0, 0, 0] # Format [x1, y1, x2, y2, confidence]
            best_eye_data = [0, 0, 0, 0, 0] # L and R eye treated as the same

            boxes = r.boxes
            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
                confidence = math.ceil((box.conf[0]*100))/100
                class_index = int(box.cls[0])

                # Get the best found eye data by compare confidence
                if class_index == 0:
                    if best_iris_data[4] < confidence:
                        best_iris_data = [x1, y1, x2, y2, confidence]
                if class_index in [1, 2]:
                    if best_eye_data[4] < confidence:
                        best_eye_data = [x1, y1, x2, y2, confidence]

            # Draw boxes for best found iris and eye data
            x1, y1, x2, y2, confidence = best_iris_data
            draw_box_data(img, x1, y1, x2, y2, confidence, "Iris", class_color[0])
            x1, y1, x2, y2, confidence = best_eye_data
            draw_box_data(img, x1, y1, x2, y2, confidence, "Eye", class_color[1])
            
            # TODO: Predict gaze direction
            cv2.putText(img, "right ", [50, 50], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


def draw_box_data(img, x1, y1, x2, y2, confidence, text, color):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, "{0} {1}".format(text, confidence), [x1, y1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


if __name__ == '__main__':
    predict_webcam()