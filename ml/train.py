"""Train the AgriAssist CNN-based YOLOv8 rice-pest detector.

Expected dataset layout:
  dataset/rice-pests/data.yaml
  dataset/rice-pests/train/images + labels
  dataset/rice-pests/valid/images + labels
  dataset/rice-pests/test/images + labels

Example:
  python ml/train.py --data dataset/rice-pests/data.yaml --epochs 50
"""
import argparse
import os
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, help="Path to YOLO data.yaml")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--model", default="yolov8n.pt")
parser.add_argument("--device", default="0", help="0 for NVIDIA GPU, cpu for CPU")
parser.add_argument("--project", default="runs/agriassist")
parser.add_argument("--name", default="rice_pest_cnn")
args = parser.parse_args()

model = YOLO(args.model)
model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
            device=args.device, project=args.project, name=args.name, patience=15,
            pretrained=True, workers=2, verbose=True)

best = os.path.join(args.project, args.name, "weights", "best.pt")
out = os.path.join(os.path.dirname(__file__), "weights")
os.makedirs(out, exist_ok=True)
if os.path.exists(best):
    import shutil
    shutil.copy2(best, os.path.join(out, "best.pt"))
    print(f"Saved trained model to {os.path.join(out, 'best.pt')}")
else:
    print("Training completed but best.pt was not found at the expected path.")
