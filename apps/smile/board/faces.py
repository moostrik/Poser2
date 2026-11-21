from __future__ import annotations

from threading import Lock
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from ..face import FaceObservations


class HasFaces(Protocol):
    """The faces the face finder judged last, in the view, with their camera frame's time."""
    def get_faces(self) -> FaceObservations | None: ...
    def set_faces(self, faces: FaceObservations) -> None: ...


class FaceStoreMixin:
    """Thread-safe storage of the newest judged faces."""

    def __init__(self) -> None:
        self._faces_lock = Lock()
        self._faces: FaceObservations | None = None

    def get_faces(self) -> FaceObservations | None:
        with self._faces_lock:
            return self._faces

    def set_faces(self, faces: FaceObservations) -> None:
        with self._faces_lock:
            self._faces = faces
