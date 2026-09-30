"""One detected face turned into one mark of the operator overlay: its box, colour and label.
No GL: arithmetic and text, built once per frame for the FaceOverlayLayer (after poser2's marks)."""

from dataclasses import dataclass

from modules.face.FaceFinder import FaceFrame
from modules.face.quality import FaceObservation, Rejection, REJECTION_LABELS

Color = tuple[float, float, float, float]

SUBJECT_COLOR: Color = (0.2, 1.0, 0.2, 1.0)     # the face the sequence uses
ACCEPTED_COLOR: Color = (1.0, 1.0, 1.0, 0.9)    # qualifies, but a larger face is the subject
REJECTED_COLOR: Color = (0.6, 0.6, 0.6, 0.8)    # did not pass a check
LABEL_BG: Color = (0.0, 0.0, 0.0, 0.6)

SUBJECT_LINE_PX: float = 3.0
LINE_PX: float = 1.5


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
    """The readout of a qualifying face; the failed check with its value of a rejected one."""
    p = face.pose
    if face.rejection is None:
        eyes: str = '-' if face.eyes is None else f'{face.eyes:.2f}'
        return f'q{face.quality:.2f} yaw {p.yaw:.0f} pitch {p.pitch:.0f} roll {p.roll:.0f} eyes {eyes}'
    value: str = {
        Rejection.LOW_SCORE: f'{face.score:.2f}',
        Rejection.SMALL:     f'{face.size:.2f}',
        Rejection.TURNED:    f'{p.yaw:.0f}',
        Rejection.PITCHED:   f'{p.pitch:.0f}',
        Rejection.TILTED:    f'{p.roll:.0f}',
        Rejection.EYES:      '-' if face.eyes is None else f'{face.eyes:.2f}',
    }[face.rejection]
    return f'{REJECTION_LABELS[face.rejection]} {value}'


def build_marks(frame: FaceFrame) -> list[Mark]:
    """A mark per face: rejected first, under everyone; the subject last, drawn over all."""
    marks: list[Mark] = []
    for face in frame.faces:
        x0, y0, x1, y1 = face.bbox
        if face.rejection is not None:
            color = REJECTED_COLOR
        elif face.subject:
            color = SUBJECT_COLOR
        else:
            color = ACCEPTED_COLOR
        # the subject is thick even when rejected: it is still the face a capture would use
        line: float = SUBJECT_LINE_PX if face.subject else LINE_PX
        marks.append(Mark(x0, y0, x1 - x0, y1 - y0, face.kps, color, line, face_label(face)))
    order: list[int] = [i for i, f in enumerate(frame.faces)]
    order.sort(key=lambda i: (frame.faces[i].rejection is None, frame.faces[i].subject))
    return [marks[i] for i in order]
