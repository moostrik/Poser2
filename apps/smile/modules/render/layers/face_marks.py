"""One detected face turned into one mark: its box, colour and label. No GL: arithmetic and text
(after poser2's marks). `build_marks` marks every face of a detector frame (a rejected one with its
reason); `SubjectMark` is the chosen face's box, kept while its charge drains: the box the camera
turns to colour, and its mark when the face has left."""

from dataclasses import dataclass

from modules.face.FaceFinder import FaceFrame
from modules.face.quality import FaceObservation, rejection_label

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
    dots: Color | None = None                   # colour of the keypoints; None: as the outline


def mix(a: Color, b: Color, t: float) -> Color:
    """From a (t 0) to b (t 1)."""
    t = min(max(t, 0.0), 1.0)
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t, a[3] + (b[3] - a[3]) * t)


def build_marks(frame: FaceFrame, subject_color: Color = SUBJECT_COLOR, charge: float = 0.0) -> list[Mark]:
    """A mark per face: rejected first, under everyone; the subject last, drawn over all. A rejected
    face is labelled with the failed check, its value and the limit; the panel has the readout.
    The subject's outline goes from grey to its colour with the charge, its dots are grey."""
    marks: list[Mark] = []
    for face in frame.faces:
        x0, y0, x1, y1 = face.bbox
        label: str = rejection_label(face) if face.rejection is not None else ''
        dots: Color | None = None
        if face.subject:
            # also when rejected: it is still the face a capture would use, and its charge drains
            color = mix(REJECTED_COLOR, subject_color, charge)
            dots = REJECTED_COLOR
        elif face.rejection is not None:
            color = REJECTED_COLOR
        else:
            color = ACCEPTED_COLOR
        line: float = SUBJECT_LINE_PX if face.subject else LINE_PX
        marks.append(Mark(x0, y0, x1 - x0, y1 - y0, face.kps, color, line, label, dots))
    order: list[int] = [i for i, f in enumerate(frame.faces)]
    order.sort(key=lambda i: (frame.faces[i].rejection is None, frame.faces[i].subject))
    return [marks[i] for i in order]


class SubjectMark:
    """The chosen face's mark, without a label: the subject's box, its outline going from grey to the active
    colour with the charge and its dots grey. While the charge drains it stays on the face that is
    there, or else where it last was; None once the charge is empty."""

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
        color: Color = mix(REJECTED_COLOR, active_color, charge)
        return Mark(x0, y0, x1 - x0, y1 - y0, kps, color, SUBJECT_LINE_PX, '', REJECTED_COLOR)
