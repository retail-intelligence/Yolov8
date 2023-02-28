from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.pt", task='detect')  # load an official model
#model = YOLO("path/to/best.pt")  # load a custom model

# Predict with the model
results = model.track("./data/video/video.mp4")  # predict on an image