"""AgriAssist CNN-based rice-pest object detection.

The dataset supplied by the project is YOLOv8 object-detection data. Ultralytics
YOLOv8 uses a convolutional neural network (CNN) based detection architecture.
The trained best.pt is produced by ml/train.py and is required for real
predictions. No fake/deterministic prediction is returned when the model is
missing.
"""
import os
from typing import Any

MODEL_PATH = os.getenv("AGRIASSIST_MODEL", os.path.join(os.path.dirname(__file__), "weights", "best.pt"))
CLASS_INFO = {
    "brown-planthopper": "Brown Planthopper",
    "green-leafhopper": "Green Leafhopper",
    "leaf-folder": "Leaf Folder",
    "rice-bug": "Rice Bug",
    "stem-borer": "Stem Borer",
    "whorl-maggot": "Whorl Maggot",
}


def _load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Trained CNN model not found. Put the trained model at ml/weights/best.pt "
            "or set AGRIASSIST_MODEL. Run ml/train.py using the supplied rice-pest dataset."
        )
    from ultralytics import YOLO
    return YOLO(MODEL_PATH)


def predict_image(path: str) -> dict[str, Any]:
    model = _load_model()
    results = model.predict(source=path, conf=0.25, iou=0.45, verbose=False)
    result = results[0]
    detections = []
    names = result.names or {}
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = [round(float(v), 2) for v in box.xyxy[0].tolist()]
            raw = str(names.get(cls_id, cls_id)).strip().lower().replace("_", "-").replace(" ", "-")
            label = CLASS_INFO.get(raw, str(names.get(cls_id, raw)))
            detections.append({"slug": raw, "label": label, "confidence": round(conf * 100, 2), "box": xyxy})
    detections.sort(key=lambda x: x["confidence"], reverse=True)
    if not detections:
        return {"mode": "cnn-yolov8", "detected": False, "detections": [], "slug": None, "label": "No supported rice pest detected", "confidence": 0.0}
    top = detections[0]
    return {"mode": "cnn-yolov8", "detected": True, "detections": detections, "slug": top["slug"], "label": top["label"], "confidence": top["confidence"]}
