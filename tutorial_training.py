from ultralytics import YOLO
from sys import platform

def train():
    model = YOLO("yolov8m.pt")
    model.train(data="datasets/data.yaml", epochs=20)
    model.val()

if __name__ == '__main__':
    train()