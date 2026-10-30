"""Export the YOLOv8 person detector to ONNX for the replay backend's host detector.

The same weights the device blobs were built from (`data/models/base/yolov8n.pt`), at the wide
blob's input size, with a dynamic batch so one run takes every camera's frame of a round
(ultralytics makes the height and width symbols too; the size stays in the file's `imgsz`
metadata, which the detector reads).
ultralytics is not a dependency of the apps: run this once with it added for the call.

Usage (from the repo root, on a PC with the venv):
    uv run --with ultralytics python tools/models/export_yolo_to_onnx.py
    uv run --with ultralytics python tools/models/export_yolo_to_onnx.py --weights data/models/base/yolov8n.pt --output data/models/yolov8n_coco_512x448.onnx --width 512 --height 448

The output is committed under data/models (LFS) like the pose models; the detector reads the
input size off the file.
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def export(weights: str, output: str, width: int, height: int, opset: int) -> None:
    model = YOLO(weights)
    # ultralytics writes <weights stem>.onnx next to the weights; dynamic = batch, height and width
    exported = model.export(format='onnx', imgsz=(height, width), dynamic=True, simplify=True, opset=opset, half=False)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported), output)
    print(f"Exported {weights} to {output} at {width}x{height} (in the imgsz metadata), opset {opset}, dynamic batch")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--weights', default='data/models/base/yolov8n.pt')
    parser.add_argument('--output', default='data/models/yolov8n_coco_512x448.onnx')
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=448)
    parser.add_argument('--opset', type=int, default=17)
    args = parser.parse_args()
    export(args.weights, args.output, args.width, args.height, args.opset)
