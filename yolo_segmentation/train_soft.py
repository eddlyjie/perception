from ultralytics import YOLO
from roboflow import Roboflow
rf = Roboflow(api_key="Ynrb4dBR6s833satPwMF")
project = rf.workspace("kevinws-95dtd").project("wbcd-cloth")
version = project.version(4)
dataset = version.download("yolov8")
model = YOLO("yolov8m-seg.pt")       
results = model.train(data="wbcd-cloth-4/data.yaml", imgsz=640, batch=12, epochs=60, plots=True)
