from ultralytics import YOLO
from roboflow import Roboflow
rf = Roboflow(api_key="Ynrb4dBR6s833satPwMF")
project = rf.workspace("kevinws-95dtd").project("wbcd-multi-object-keypoint")
version = project.version(3)
dataset = version.download("yolov8")
model = YOLO("yolo11s-pose")           
model.train(data="WBCD-Multi-Object-Keypoint-3/data.yaml", imgsz=640, batch=12, epochs=60, plots=True)