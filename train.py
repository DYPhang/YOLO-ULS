import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8s-uls.yaml')
    model.train(data='ultralytics/dataset/data.yaml',
                cache=False,
                project='runs',
                name='exp',
                epochs=200,
                batch=8,
                optimizer='SGD',
                imgsz=640,
                # resume='runs/train/exp/weights/last.pt',
                )
