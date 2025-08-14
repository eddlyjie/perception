from ultralytics import YOLO
from roboflow import Roboflow
rf = Roboflow(api_key="WQKWXokKPdImFLLLxs5R")
project = rf.workspace("box-gatzq").project("lolipop")
version = project.version(2)
dataset = version.download("yolov8")                
model = YOLO("yolov8m-seg.pt")       
results = model.train(data="lolipop-2/data.yaml", imgsz=640, batch=12, epochs=60, plots=True)
