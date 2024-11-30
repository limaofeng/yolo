from ultralytics import YOLO

import os
import sys
project_path = os.getcwd()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load a model
# model = YOLO("yolo11n.yaml")  # build a new model from YAML
# model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11n.yaml").load( project_path + "/weights/yolo11n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="configs/plate.yaml", epochs=100, imgsz=640, device=[0,1])

# 导出权重
model.save(project_path + "/weights/custom_yolo.pt")