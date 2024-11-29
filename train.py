from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n.yaml")  # build a new model from YAML
# model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="plate.yaml", epochs=100, imgsz=640, device="0")

# 导出权重
model.save("weights/custom_yolo.pt")