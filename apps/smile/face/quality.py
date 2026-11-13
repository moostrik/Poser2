"""Does a detected face qualify for a portrait? Its score, size, the portrait crop's place, its head pose and its eyes
(measured by `modules.inference.face`) against the gates' limits; the first failing check is the face's rejection.
Pure arithmetic: no model, no thread, no settings object, so it is testable with synthetic values."""

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Callable

from modules.inference.face import HeadPose


class Rejection(IntEnum):
    """Why a face does not qualify; the first failing check, in this order."""
    LOW_SCORE = auto()  # detector score below confidence
    SMALL     = auto()  # eye distance below eye_min: too few pixels for the portrait
    OUTSIDE   = auto()  # the portrait crop nearer the image edge than crop_margin: too close, or too near an edge
    TURNED    = auto()  # yaw past max_yaw
    PITCHED   = auto()  # pitch past max_pitch
    TILTED    = auto()  # roll past max_roll
    EYES      = auto()  # an eye less open than eye_open


@dataclass(frozen=True)
class FaceObservation:
    """One face of one frame, as the overlay shows it. Coordinates are normalised to the image."""
    bbox: tuple[float, float, float, float]     # x0, y0, x1, y1
    crop: tuple[float, float, float, float]     # the portrait crop as an upright box (past the image as far as crop_margin allows)
    kps: tuple[tuple[float, float], ...]        # eyes, nose, mouth corners
    score: float
    eye_dist: float                             # distance between the eyes relative to the image width
    edge: float                                 # distance of the portrait crop from the nearest image edge, relative to the crop width; negative past it
    pose: HeadPose
    eyes: float | None                          # openness of the less open eye, None if not measured
    rejection: Rejection | None
    limit: float                                # the setting the failed check compared to; 0 when accepted
    distance: float                             # how far from qualifying: 0 good .. 1 far (see distance())
    subject: bool                               # the face the sequence uses

    @property
    def accepted(self) -> bool:
        return self.rejection is None


@dataclass(frozen=True)
class FaceObservations:
    """Every face of one camera frame as judged, for the overlay and the sequencer's charge."""
    faces: tuple[FaceObservation, ...]
    timestamp: float     # when the camera frame arrived, not when it was judged: a camera that stopped goes stale

    @property
    def subject(self) -> FaceObservation | None:
        return next((f for f in self.faces if f.subject), None)

    @property
    def qualified(self) -> bool:
        """The subject passed every check."""
        subject: FaceObservation | None = self.subject
        return subject is not None and subject.rejection is None


FaceObservationsCallback = Callable[[FaceObservations], None]


def readout(face: FaceObservation) -> str:
    """The measured values, in the panel's names: `conf 0.92 eyes 0.06 edge 0.05 yaw 4 pitch -6 roll 2 open 0.34`."""
    p: HeadPose = face.pose
    open_: str = '-' if face.eyes is None else f'{face.eyes:.2f}'
    return f'conf {face.score:.2f} eyes {face.eye_dist:.2f} edge {face.edge:.2f} yaw {p.yaw:.0f} pitch {p.pitch:.0f} roll {p.roll:.0f} open {open_}'


def rejection_label(face: FaceObservation) -> str:
    """Why the face was rejected, in the panel's names: the value measured and the limit it missed,
    e.g. `yaw 32 > 20` (an angle keeps its sign: `yaw -32 > 20` is beyond 20 the other way)."""
    p: HeadPose = face.pose
    if face.rejection is None:
        return ''
    if face.rejection == Rejection.LOW_SCORE:
        return f'conf {face.score:.2f} < {face.limit:.2f}'
    if face.rejection == Rejection.SMALL:
        return f'eyes {face.eye_dist:.2f} < {face.limit:.2f}'
    if face.rejection == Rejection.OUTSIDE:
        return f'edge {face.edge:.2f} < {face.limit:.2f}'
    if face.rejection == Rejection.TURNED:
        return f'yaw {p.yaw:.0f} > {face.limit:.0f}'
    if face.rejection == Rejection.PITCHED:
        return f'pitch {p.pitch:.0f} > {face.limit:.0f}'
    if face.rejection == Rejection.TILTED:
        return f'roll {p.roll:.0f} > {face.limit:.0f}'
    open_: str = '-' if face.eyes is None else f'{face.eyes:.2f}'
    return f'open {open_} < {face.limit:.2f}'


def gate(score: float, eye_dist: float, edge: float, pose: HeadPose, confidence: float, eye_min: float, crop_margin: float,
         max_yaw: float, max_pitch: float, max_roll: float) -> Rejection | None:
    """The checks before the eyes, in order; None when the face passes them all."""
    if score < confidence:
        return Rejection.LOW_SCORE
    if eye_dist < eye_min:
        return Rejection.SMALL
    if edge < crop_margin:
        return Rejection.OUTSIDE
    if abs(pose.yaw) > max_yaw:
        return Rejection.TURNED
    if abs(pose.pitch) > max_pitch:
        return Rejection.PITCHED
    if abs(pose.roll) > max_roll:
        return Rejection.TILTED
    return None


# a rejected face is never closer to good than this: crossing a limit shows as a step
REJECTED_FLOOR: float = 0.2


def distance(score: float, eye_dist: float, edge: float, pose: HeadPose, eyes: float | None, confidence: float, eye_min: float,
             crop_margin: float, max_yaw: float, max_pitch: float, max_roll: float, eye_open: float, falloff: float) -> float:
    """How far a face is from qualifying: 0 when it passes every check, REJECTED_FLOOR just past a
    limit, 1 when its worst check is `falloff` times the limit past it (1: twice a maximum, or zero
    for a minimum; for the crop, `falloff` crop widths short of the margin). The eyes count only when
    they were measured."""
    def below(value: float, limit: float) -> float:
        return (limit - value) / max(limit, 1e-6)

    def above(value: float, limit: float) -> float:
        return (value - limit) / max(limit, 1e-6)

    excess: list[float] = [below(score, confidence), below(eye_dist, eye_min), crop_margin - edge,
 above(abs(pose.yaw), max_yaw), above(abs(pose.pitch), max_pitch), above(abs(pose.roll), max_roll)]
    if eyes is not None:
        excess.append(below(eyes, eye_open))
    worst: float = max(excess)
    if worst <= 0.0:
        return 0.0
    return REJECTED_FLOOR + (1.0 - REJECTED_FLOOR) * min(worst / max(falloff, 1e-6), 1.0)
