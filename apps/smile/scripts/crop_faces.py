"""Crop a photo as the face finder crops a capture: the largest face, the eyes levelled, at the portrait size.
For testing any photo in an image generator that starts from a cropped portrait (SMILE's test workflow, for one).

    .venv\\Scripts\\python apps\\smile\\scripts\\crop_faces.py photo.jpg [more photos] [-o folder] [--zoom 1.2] [--offset 0.0] [--size 1024 1280]

Writes <name>_crop.png next to each photo (or into -o). The zoom and offset default to the face finder's
defaults (the panel's Portrait zoom and offset); pass the preset's values to match a capture exactly.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import torch  # noqa: E402

from modules.camera import CameraFrame, FrameType  # noqa: E402
from modules.inference import GpuImage, ImageUpload  # noqa: E402
from modules.inference.face import FaceDetector, FaceDetectorSettings  # noqa: E402
from apps.smile.face import FaceFinderSettings, portrait_crop  # noqa: E402


def crop_file(detector: FaceDetector, upload: ImageUpload, path: Path, out: Path, zoom: float, offset: float, width: int, height: int) -> str:
    image = cv2.imread(str(path))
    if image is None:
        return f'{path}: not an image'
    frame = CameraFrame(0, FrameType.VIDEO, image, time.perf_counter())
    images, ready = upload.upload([frame], torch.cuda.current_stream())
    faces = detector.detect(GpuImage(0, frame.timestamp, images[0], ready, frame)).faces
    if not faces:
        return f'{path}: no face found'
    face = max(faces, key=lambda f: f.width * f.height)
    cv2.imwrite(str(out), portrait_crop(image, face, zoom, offset, width, height))
    return f'{path} -> {out} ({len(faces)} face{"s" if len(faces) > 1 else ""}, the largest taken)'


def main(argv: list[str] | None = None) -> int:
    defaults = FaceFinderSettings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('photos', nargs='+', type=Path)
    parser.add_argument('-o', '--out', type=Path, help='folder for the crops (default: next to each photo)')
    parser.add_argument('--zoom', type=float, default=defaults.zoom, help='portrait width relative to the face height')
    parser.add_argument('--offset', type=float, default=defaults.offset, help='vertical offset relative to the face height')
    parser.add_argument('--size', type=int, nargs=2, default=(1024, 1280), metavar=('W', 'H'), help='the crop size (the render size)')
    args = parser.parse_args(argv)
    detector = FaceDetector(FaceDetectorSettings())        # on ONNX Runtime, the models in data/models
    upload = ImageUpload()
    failed: int = 0
    for photo in args.photos:
        out: Path = (args.out / photo.name if args.out else photo).with_name(photo.stem + '_crop.png')
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
        message: str = crop_file(detector, upload, photo, out, args.zoom, args.offset, args.size[0], args.size[1])
        failed += '->' not in message
        print(message)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
