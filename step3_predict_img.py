from ultralytics import YOLO

def predict_img():
    print("Start predict image")
    # After training, find the latest train folder inside runs/detect and 
    # get the best.pt file. Some of the last train folder contains validation 
    # image so you may have to dig around a little bit
    model = YOLO("runs/detect/train9/weights/best.pt")
    # Run batched inference on a list of images
    # Create a new folder and add some image to it
    results = model(["predict_img/im2.jpg"])

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen
        result.save(filename="predict_img/result.png")  # save to disk

if __name__ == '__main__':
    predict_img()