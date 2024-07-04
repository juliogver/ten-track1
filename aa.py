from ultralytics import YOLO

model = YOLO('yolov8x')

result = model.track ('input_videos/Y2meta.app-Best Points of February �� Tracked by the SwingVision app-(1080p60) - Trim.mp4',save= True)


