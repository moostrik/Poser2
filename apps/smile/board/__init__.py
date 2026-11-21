"""SMILE's board: one protocol + mixin pair per slice, composed in `RenderBoard`."""

from .faces import HasFaces, FaceStoreMixin
from .capture import HasCapture, CaptureStoreMixin
from .generator_images import HasGeneratorImages, GeneratorImageStoreMixin
from .status import HasStatus, StatusStoreMixin
from .online_cell import HasOnlineCell, OnlineCell, OnlineCellStoreMixin
from .render_board import RenderBoard
