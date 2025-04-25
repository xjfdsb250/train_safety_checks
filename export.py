from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO("./runs/detect/train2/weights/best.pt")
    model.export(format='onnx', imgsz=[1024, 1024], opset=12, dynamic=False)
