"""The OAK-D's detector and tracker: the blobs and their input size."""

# https://blobconverter.luxonis.com/

from pathlib import Path

from depthai import TrackerType, OpenVINO

# The wide blobs (CAMERA.md, the OAK-D's detector): 6 shaves on a device, 7 for clips (`pipeline.model_path`).
YOLOV8_WIDE_6S: str = "yolov8n_coco_512x448_6S.blob"
YOLOV8_WIDE_7S: str = "yolov8n_coco_512x448_7S.blob"
YOLOV8_SQUARE_6S: str = "yolov8n_coco_416x416_6S.blob"
YOLOV8_SQUARE_7S: str = "yolov8n_coco_416x416_7S.blob"
DETECTOR_CONFIDENCE_THRESHOLD: float = 0.5
DETECTOR_OVERLAP_THRESHOLD: float = 0.5

TRACKER_PERSON_LABEL: int = 0
TRACKER_TYPE: TrackerType = TrackerType.ZERO_TERM_IMAGELESS     # COLOR_HISTOGRAM is more accurate but can drift when it loses a player


def detector_input_size(blob_path: str | Path) -> tuple[int, int]:
    """(width, height) the detector blob was compiled for, read off the blob (depthai stores [W, H, C, N])."""
    blob = OpenVINO.Blob(Path(blob_path))
    dims = next(iter(blob.networkInputs.values())).dims
    return int(dims[0]), int(dims[1])
