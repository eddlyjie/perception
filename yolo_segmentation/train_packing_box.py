from ultralytics import YOLO
from roboflow import Roboflow
rf = Roboflow(api_key="WQKWXokKPdImFLLLxs5R")
project = rf.workspace("box-gatzq").project("box-6ifyp")
version = project.version(5)
dataset = version.download("yolov8")
model = YOLO("yolov8m-seg.pt")       
results = model.train(data="box-5/data.yaml", imgsz=640, batch=12, epochs=60, plots=True)