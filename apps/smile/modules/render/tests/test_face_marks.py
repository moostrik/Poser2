"""Tests for the face overlay's marks: colours, labels and draw order (no GL needed)."""

import unittest

from modules.face.FaceFinder import FaceFrame
from modules.face.quality import FaceObservation, Pose, Rejection
from modules.render.layers.face_marks import ACCEPTED_COLOR, REJECTED_COLOR, SUBJECT_COLOR, SUBJECT_LINE_PX, LINE_PX, build_marks, face_label


def observation(rejection: Rejection | None = None, subject: bool = False, eyes: float | None = 0.3, bbox=(0.1, 0.2, 0.5, 0.7)) -> FaceObservation:
    return FaceObservation(bbox=bbox, kps=((0.2, 0.3),) * 5, score=0.9, size=0.3, pose=Pose(4.4, -6.0, 2.0),
                           eyes=eyes, quality=0.84, rejection=rejection, subject=subject)


class TestFaceLabel(unittest.TestCase):
    def test_an_accepted_face_reads_out_its_values(self):
        self.assertEqual(face_label(observation()), 'q0.84 yaw 4 pitch -6 roll 2 eyes 0.30')

    def test_a_rejected_face_shows_the_reason_and_its_value(self):
        self.assertEqual(face_label(observation(Rejection.TURNED, eyes=None)), 'turned 4')
        self.assertEqual(face_label(observation(Rejection.EYES, eyes=0.12)), 'eyes 0.12')
        self.assertEqual(face_label(observation(Rejection.SMALL, eyes=None)), 'small 0.30')


class TestBuildMarks(unittest.TestCase):
    def test_colours_and_lines(self):
        frame = FaceFrame((observation(subject=True), observation(), observation(Rejection.EYES)), True, 0.0)
        marks = {m.color: m for m in build_marks(frame)}
        self.assertEqual(marks[SUBJECT_COLOR].line, SUBJECT_LINE_PX)
        self.assertEqual(marks[ACCEPTED_COLOR].line, LINE_PX)
        self.assertEqual(marks[REJECTED_COLOR].line, LINE_PX)

    def test_a_rejected_subject_is_grey_but_thick(self):
        marks = build_marks(FaceFrame((observation(Rejection.TURNED, subject=True),), False, 0.0))
        self.assertEqual(marks[0].color, REJECTED_COLOR)
        self.assertEqual(marks[0].line, SUBJECT_LINE_PX)

    def test_rejected_first_subject_last(self):
        frame = FaceFrame((observation(subject=True), observation(Rejection.SMALL), observation()), True, 0.0)
        colors = [m.color for m in build_marks(frame)]
        self.assertEqual(colors, [REJECTED_COLOR, ACCEPTED_COLOR, SUBJECT_COLOR])

    def test_box_from_bbox(self):
        mark = build_marks(FaceFrame((observation(bbox=(0.1, 0.2, 0.5, 0.7)),), False, 0.0))[0]
        self.assertAlmostEqual(mark.x, 0.1)
        self.assertAlmostEqual(mark.y, 0.2)
        self.assertAlmostEqual(mark.w, 0.4)
        self.assertAlmostEqual(mark.h, 0.5)


if __name__ == '__main__':
    unittest.main()
