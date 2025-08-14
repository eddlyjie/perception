from ultralytics import YOLO
from roboflow import Roboflow
rf = Roboflow(api_key="ebdr2icwLiOioiWpflMY")
project = rf.workspace("test-z2ldp").project("seg_cube")
version = project.version(1)
dataset = version.download("yolov8")
model = YOLO("yolov8m-seg.pt")       
results = model.train(data="seg_cube-1/data.yaml", imgsz=640, batch=12, epochs=60, plots=True)
                