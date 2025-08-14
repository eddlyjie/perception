from roboflow import Roboflow
from ultralytics import YOLO
rf = Roboflow(api_key="Ynrb4dBR6s833satPwMF")
project = rf.workspace("kevinws-95dtd").project("wbcd-multi-object")
version = project.version(18)
dataset = version.download("yolov8")
model = YOLO("yolov8m-seg.pt")       
results = model.train(data="WBCD-Multi-Object-18/data.yaml", imgsz=640, batch=10, epochs=100, plots=True)

