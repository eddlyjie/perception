from ultralytics import YOLO
from roboflow import Roboflow
rf = Roboflow(api_key="Ynrb4dBR6s833satPwMF")
project = rf.workspace("kevinws-95dtd").project("wbcd-lollipop")
version = project.version(7)
dataset = version.download("yolov8")
model = YOLO("yolov8m-seg.pt")       
results = model.train(data="WBCD-lollipop-7/data.yaml", imgsz=640, batch=12, epochs=100, plots=True)
                