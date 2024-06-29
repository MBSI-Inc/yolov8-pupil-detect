from ultralytics import YOLO

def train():
    # This will create a new model from a preset (this preset is yolov8 medium weight)
    model = YOLO("yolov8m.pt")
    # This is the location of the dataset. Epochs is the number of time we spent on training.
    # Generally longer is better, but not too much.
    model.train(data="datasets/data.yaml", epochs=20)
    # Start the model valuation. Technically it's not required for model training, but
    # it's important to know the training result.
    model.val()

if __name__ == '__main__':
    train()