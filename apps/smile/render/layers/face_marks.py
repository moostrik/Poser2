"""One detected face turned into one mark: its box (the portrait crop's, upright), colour and label. No GL: arithmetic and text
(after poser2's marks). `build_marks` marks every face of a detector frame (a rejected one with its
reason); `SubjectMark` is the chosen face's box, kept while its charge drains: the box the camera
turns to colour, and its mark when the face has left."""

from dataclasses import dataclass

from modules.face import FaceFrame, FaceObservation, rejection_label

Color = tuple[float, float, float, float]

SUBJECT_COLOR: Color = (0.2, 1.0, 0.2, 1.0)     # the face the sequence uses
ACCEPTED_COLOR: Color = (1.0, 1.0, 1.0, 0.9)    # qualifies, but a larger face is the subject
REJECTED_COLOR: Color = (0.6, 0.6, 0.6, 0.8)    # did not pass a check
LABEL_BG: Color = (0.0, 0.0, 0.0, 0.6)

SUBJECT_LINE_PX: float = 3.0
LINE_PX: float = 1.5
BOX_PAD_PX: float = 8.0     # a box that reaches past the image is drawn this far inside its edge

# no marks from faces older than this: the detector only runs in the camera states
FACES_STALE: float = 0.5

Box = tuple[float, float, float, float]         # x0, y0, x1, y1, normalised
Pad = tuple[float, float]                       # of the image's width and height
NO_PAD: Pad = (0.0, 0.0)


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
    subject: bool = False                       # the chosen face
    far: float = 0.0                            # its distance from good, 0..1


def mix(a: Color, b: Color, t: float) -> Color:
    """From a (t 0) to b (t 1)."""
    t = min(max(t, 0.0), 1.0)
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t, a[3] + (b[3] - a[3]) * t)


def fit_box(box: Box, pad: Pad = NO_PAD) -> Box:
    """The box clamped to the image, `pad` inside its edge: an edge past that line is set to it, so a crop
    that reaches past the image (as far as the face's crop_margin allows) is drawn whole, over the face.
    An edge that would cross the other is held at it (a width of 0: no box)."""
    x0: float = max(box[0], pad[0])
    y0: float = max(box[1], pad[1])
    return (x0, y0, max(min(box[2], 1.0 - pad[0]), x0), max(min(box[3], 1.0 - pad[1]), y0))


def face_fields(frame: FaceFrame, pad: Pad = NO_PAD) -> list[tuple[float, float, float, float, float]]:
    """Per face its box (`pad` inside the image) and distance from good (x, y, w, h, far): the fields the camera shows."""
    fields: list[tuple[float, float, float, float, float]] = []
    for face in frame.faces:
        x0, y0, x1, y1 = fit_box(face.crop, pad)
        fields.append((x0, y0, x1 - x0, y1 - y0, face.distance))
    return fields


def build_marks(frame: FaceFrame, subject_color: Color = SUBJECT_COLOR, charge: float = 0.0, pad: Pad = NO_PAD) -> list[Mark]:
    """A mark per face (its box `pad` inside the image): rejected first, under everyone; the subject
    last, drawn over all. A rejected face is labelled with the failed check, its value and the limit;
    the panel has the readout. The subject's outline goes from grey to its colour with the charge, its
    dots are grey."""
    marks: list[Mark] = []
    for face in frame.faces:
        x0, y0, x1, y1 = fit_box(face.crop, pad)
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
        marks.append(Mark(x0, y0, x1 - x0, y1 - y0, face.kps, color, line, label, dots, face.subject, face.distance))
    order: list[int] = [i for i, f in enumerate(frame.faces)]
    order.sort(key=lambda i: (frame.faces[i].rejection is None, frame.faces[i].subject))
    return [marks[i] for i in order]


class SubjectMark:
    """The chosen face's mark, without a label: the subject's box, its outline going from grey to the active
    colour with the charge and its dots grey. While the charge drains it stays on the face that is
    there, or else where it last was (then as far from good as can be); None once the charge is empty."""

    def __init__(self) -> None:
        self._box: tuple[Box, tuple[tuple[float, float], ...]] | None = None
        self._far: float = 1.0

    def update(self, frame: FaceFrame | None, now: float, charge: float, active_color: Color = SUBJECT_COLOR, pad: Pad = NO_PAD) -> Mark | None:
        """The mark for this frame, its box `pad` inside the image."""
        subject: FaceObservation | None = None
        if frame is not None and now - frame.timestamp <= FACES_STALE:
            subject = frame.subject
        self._far = 1.0
        if subject is not None:
            self._box = (fit_box(subject.crop, pad), subject.kps)
            self._far = subject.distance

        qualifies: bool = subject is not None and subject.rejection is None
        if not qualifies and charge <= 0.0:
            self._box = None
        if self._box is None:
            return None
        (x0, y0, x1, y1), kps = self._box
        color: Color = mix(REJECTED_COLOR, active_color, charge)
        return Mark(x0, y0, x1 - x0, y1 - y0, kps, color, SUBJECT_LINE_PX, '', REJECTED_COLOR, True, self._far)
