from ultralytics import YOLO

# Load a model
model = YOLO("weights/custom_yolo.pt")

# Perform object detection on an image
results = model("./1732584840702.jpg")

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk