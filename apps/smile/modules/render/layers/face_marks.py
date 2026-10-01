"""One detected face turned into one mark: its box, colour and label. No GL: arithmetic and text
(after poser2's marks). `build_marks` is the operator overlay (every face, with its readout);
`SubjectMark` is the show's own mark, the green box of the chosen face."""

from dataclasses import dataclass

from modules.face.FaceFinder import FaceFrame
from modules.face.quality import FaceObservation, readout, rejection_label

Color = tuple[float, float, float, float]

SUBJECT_COLOR: Color = (0.2, 1.0, 0.2, 1.0)     # the face the sequence uses
ACCEPTED_COLOR: Color = (1.0, 1.0, 1.0, 0.9)    # qualifies, but a larger face is the subject
REJECTED_COLOR: Color = (0.6, 0.6, 0.6, 0.8)    # did not pass a check
LABEL_BG: Color = (0.0, 0.0, 0.0, 0.6)

SUBJECT_LINE_PX: float = 3.0
LINE_PX: float = 1.5

# no marks from faces older than this: the detector only runs in the camera states
FACES_STALE: float = 0.5


@dataclass(frozen=True)
class Mark:
    x: float                                    # normalised box, top left origin
    y: float
    w: float
    h: float
    kps: tuple[tuple[float, float], ...]        # normalised keypoints
    color: Color
    line: float                                 # outline width in pixels
    label: str


def face_label(face: FaceObservation) -> str:
    """The readout of a qualifying face; the failed check, its value and the limit of a rejected one."""
    if face.rejection is not None:
        return rejection_label(face)
    return readout(face)


def build_marks(frame: FaceFrame, subject_color: Color = SUBJECT_COLOR) -> list[Mark]:
    """A mark per face: rejected first, under everyone; the subject last, drawn over all."""
    marks: list[Mark] = []
    for face in frame.faces:
        x0, y0, x1, y1 = face.bbox
        if face.rejection is not None:
            color = REJECTED_COLOR
        elif face.subject:
            color = subject_color
        else:
            color = ACCEPTED_COLOR
        # the subject is thick even when rejected: it is still the face a capture would use
        line: float = SUBJECT_LINE_PX if face.subject else LINE_PX
        marks.append(Mark(x0, y0, x1 - x0, y1 - y0, face.kps, color, line, face_label(face)))
    order: list[int] = [i for i, f in enumerate(frame.faces)]
    order.sort(key=lambda i: (frame.faces[i].rejection is None, frame.faces[i].subject))
    return [marks[i] for i in order]


class SubjectMark:
    """The show's mark, without a label: the subject's box in the active colour while it qualifies
    (its charge rises); in grey while the charge drains, on the face that is there or else where it
    last was; None once the charge is empty."""

    def __init__(self) -> None:
        self._box: tuple[tuple[float, float, float, float], tuple[tuple[float, float], ...]] | None = None

    def update(self, frame: FaceFrame | None, now: float, charge: float, active_color: Color = SUBJECT_COLOR) -> Mark | None:
        subject: FaceObservation | None = None
        if frame is not None and now - frame.time <= FACES_STALE:
            subject = frame.subject
        if subject is not None:
            self._box = (subject.bbox, subject.kps)

        qualifies: bool = subject is not None and subject.rejection is None
        if not qualifies and charge <= 0.0:
            self._box = None
        if self._box is None:
            return None
        (x0, y0, x1, y1), kps = self._box
        color: Color = active_color if qualifies else REJECTED_COLOR
        return Mark(x0, y0, x1 - x0, y1 - y0, kps, color, SUBJECT_LINE_PX, '')
