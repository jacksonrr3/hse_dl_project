from ultralytics import YOLO

model = YOLO("yolo11m.pt")

model.train(data="../dataset/yolo/data.yaml", epochs=1)