"""The OAK4's models, fetched once from Luxonis's zoo into `data/models`.

    python -m modules.camera.oak4.models [--models data/models]

The chip takes a model compiled by Luxonis, which the zoo hands out as an NN archive (a `.tar.xz`
with the compiled model and the config that says what its outputs mean). Public models need no
key. The archive is copied under the name the backend expects (`definitions.POSE_ARCHIVE`); it is
not in git (`data/models/*.tar.xz` is ignored until the models are settled), so every OAK4 PC runs
this once, with internet, and again for a new model.

`definitions.PERSON_DETECTOR_ARCHIVE` is not fetched here: the zoo has no 5:4 size for this chip,
so it is compiled from `data/models/base/yolov8n.pt` through Luxonis's HubAI, at 640x512, with
`hubai-sdk` and a Hub API key, and copied to `data/models` by hand.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import depthai as dai

from .definitions import POSE_MODEL, POSE_ARCHIVE, DETECTOR_MODEL, DETECTOR_ARCHIVE, TOP_DOWN_MODEL, TOP_DOWN_ARCHIVE

logger = logging.getLogger(__name__)

MODELS: dict[str, str] = {              # zoo slug -> file name in data/models
    POSE_MODEL: POSE_ARCHIVE,
    DETECTOR_MODEL: DETECTOR_ARCHIVE,
    TOP_DOWN_MODEL: TOP_DOWN_ARCHIVE,
}


def fetch(models: Path) -> int:
    """Every model in `MODELS` into `models`; returns the count that failed."""
    failed: int = 0
    for slug, name in MODELS.items():
        target: Path = models / name
        try:
            description = dai.NNModelDescription(slug, platform='RVC4')
            path = Path(dai.getModelFromZoo(description, useCached=True, progressFormat='pretty'))
            if path.is_dir():       # a cache that holds the archive unpacked beside it
                path = next(path.glob('*.tar.xz'))
            shutil.copyfile(path, target)
            logger.info(f'{slug} -> {target} ({target.stat().st_size / 1e6:.1f} MB)')
        except Exception as exc:
            logger.error(f'{slug} not fetched: {exc}')
            failed += 1
    return failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="fetch the OAK4's models from the Luxonis zoo into data/models")
    parser.add_argument('--models', default='data/models', help='the folder, the backend\'s model_path')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    return fetch(Path(args.models))


if __name__ == '__main__':
    sys.exit(main())
