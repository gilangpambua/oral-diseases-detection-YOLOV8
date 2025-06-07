from pathlib import Path
import sys

file_path = Path(__file__).resolve()

root_path = file_path.parent

if root_path not in sys.path:
    sys.path.append(str(root_path))

ROOT = root_path

SOURCES_LIST = ["Image"]

DETECTION_MODEL_DIR = ROOT / 'weights' / 'detection'
YOLOv8n = DETECTION_MODEL_DIR / "Best_YOLOV8N.pt"
YOLOv8s = DETECTION_MODEL_DIR / "Best_YOLOV8S.pt"
YOLOv8m = DETECTION_MODEL_DIR / "Best_YOLOV8M.pt"
YOLOv8nn = DETECTION_MODEL_DIR / "bestN.pt"
YOLOv8sn = DETECTION_MODEL_DIR / "bestS.pt"
YOLOv8mn = DETECTION_MODEL_DIR / "bestM.pt"

DETECTION_MODEL_LIST = [
    "Best_YOLOV8N.pt",
    "Best_YOLOV8S.pt",
    "Best_YOLOV8M.pt",
    "bestN.pt",
    "bestS.pt",
    "bestM.pt"]
