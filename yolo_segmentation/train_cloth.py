from ultralytics import YOLO
from roboflow import Roboflow
rf = Roboflow(api_key="WQKWXokKPdImFLLLxs5R")
project = rf.workspace("box-gatzq").project("cloth_on_top")
version = project.version(6)
dataset = version.download("yolov8")
                
model = YOLO("yolov8m-seg.pt")       
results = model.train(data="cloth_on_top-6/data.yaml", imgsz=640, batch=12, epochs=100, plots=True)
                