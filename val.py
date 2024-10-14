import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('best.pt')
    model.val(data='dataset/data.yaml',
                split='val',
                save_json=True,
                save_txt=True,
                project='runs/val',
                name='exp',
                )