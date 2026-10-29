"""Tests for the face overlay's marks: colours, labels and draw order (no GL needed)."""

import unittest

from modules.face import FaceFrame, FaceObservation, Pose, Rejection
from apps.smile.render.layers.face_marks import ACCEPTED_COLOR, FACES_STALE, REJECTED_COLOR, SUBJECT_COLOR, SUBJECT_LINE_PX, LINE_PX, SubjectMark, build_marks, face_fields, fit_box, mix


def observation(rejection: Rejection | None = None, subject: bool = False, eyes: float | None = 0.3, bbox=(0.1, 0.2, 0.5, 0.7),
                limit: float = 0.0, pose: Pose = Pose(4.4, -6.0, 2.0, 0.48), distance: float = 0.0) -> FaceObservation:
    # the marks use the portrait box (crop), not the detector's
    return FaceObservation(bbox=(0.0, 0.0, 0.0, 0.0), crop=bbox, kps=((0.2, 0.3),) * 5, score=0.9, eye_dist=0.06, edge=0.1, pose=pose,
                           eyes=eyes, rejection=rejection, limit=limit, distance=distance, subject=subject)


class TestSubjectMark(unittest.TestCase):
    def test_a_qualifying_subject_is_a_mark_without_label_and_with_grey_dots(self):
        mark = SubjectMark().update(FaceFrame((observation(), observation(subject=True)), 10.0), 10.0, 1.0)
        self.assertEqual(mark.color, SUBJECT_COLOR)
        self.assertEqual(mark.dots, REJECTED_COLOR)
        self.assertEqual(mark.label, '')
        self.assertAlmostEqual(mark.w, 0.4)

    def test_the_outline_goes_from_grey_to_the_active_colour_with_the_charge(self):
        subject = SubjectMark()
        blue = (0.0, 0.0, 1.0, 1.0)
        good = FaceFrame((observation(subject=True),), 10.0)
        away = FaceFrame((observation(Rejection.TURNED, subject=True),), 10.1)
        self.assertEqual(subject.update(good, 10.0, 0.0, blue).color, REJECTED_COLOR)
        self.assertEqual(subject.update(good, 10.0, 1.0, blue).color, blue)
        half = subject.update(good, 10.0, 0.5, blue)
        self.assertEqual(half.color, mix(REJECTED_COLOR, blue, 0.5))
        self.assertAlmostEqual(half.color[0], 0.3)
        self.assertEqual(half.dots, REJECTED_COLOR)
        # the same while the charge drains
        self.assertEqual(subject.update(away, 10.1, 0.5, blue).color, half.color)
        # the subject among all faces is marked the same
        both = FaceFrame((observation(subject=True), observation()), 0.0)
        self.assertEqual([m.color for m in build_marks(both, blue, 1.0)], [ACCEPTED_COLOR, blue])
        self.assertEqual([m.color for m in build_marks(both, blue, 0.5)], [ACCEPTED_COLOR, half.color])
        self.assertEqual([m.dots for m in build_marks(both, blue, 0.5)], [None, REJECTED_COLOR])

    def test_nothing_without_a_charge_or_a_qualifying_subject(self):
        subject = SubjectMark()
        self.assertIsNone(subject.update(None, 10.0, 0.0))
        self.assertIsNone(subject.update(FaceFrame((), 10.0), 10.0, 0.0))
        self.assertIsNone(subject.update(FaceFrame((observation(Rejection.TURNED, subject=True),), 10.1), 10.1, 0.0))

    def test_on_the_face_while_the_charge_drains(self):
        subject = SubjectMark()
        good = FaceFrame((observation(subject=True),), 10.0)
        away = FaceFrame((observation(Rejection.TURNED, subject=True, bbox=(0.3, 0.2, 0.7, 0.7)),), 10.1)
        subject.update(good, 10.0, 0.4)
        mark = subject.update(away, 10.1, 0.35)
        self.assertEqual(mark.color, mix(REJECTED_COLOR, SUBJECT_COLOR, 0.35))
        self.assertAlmostEqual(mark.x, 0.3)                       # follows the face that is there
        self.assertIsNone(subject.update(away, 10.2, 0.0))        # drained: gone

    def test_where_the_face_was_when_it_left(self):
        subject = SubjectMark()
        good = FaceFrame((observation(subject=True),), 10.0)
        subject.update(good, 10.0, 0.4)
        mark = subject.update(FaceFrame((), 10.1), 10.1, 0.3)
        self.assertEqual(mark.color, mix(REJECTED_COLOR, SUBJECT_COLOR, 0.3))
        self.assertAlmostEqual(mark.x, 0.1)
        stale = subject.update(good, 10.0 + FACES_STALE + 0.1, 0.2)     # the detector stopped: its frame is old
        self.assertAlmostEqual(stale.x, 0.1)

    def test_a_stale_detector_frame_gives_no_mark(self):
        subject = SubjectMark()
        good = FaceFrame((observation(subject=True),), 10.0)
        self.assertIsNone(subject.update(good, 20.0, 0.0))


class TestLabels(unittest.TestCase):
    @staticmethod
    def label(face: FaceObservation) -> str:
        return build_marks(FaceFrame((face,), 0.0))[0].label

    def test_a_qualifying_face_has_no_label(self):
        self.assertEqual(self.label(observation()), '')
        self.assertEqual(self.label(observation(subject=True)), '')

    def test_a_rejected_face_shows_the_check_its_value_and_the_limit(self):
        turned = observation(Rejection.TURNED, eyes=None, limit=20.0, pose=Pose(-32.4, 0.0, 0.0, 0.48))
        self.assertEqual(self.label(turned), 'yaw -32 > 20')
        self.assertEqual(self.label(observation(Rejection.EYES, eyes=0.12, limit=0.2)), 'open 0.12 < 0.20')
        self.assertEqual(self.label(observation(Rejection.SMALL, eyes=None, limit=0.08)), 'eyes 0.06 < 0.08')


class TestBuildMarks(unittest.TestCase):
    def test_colours_and_lines(self):
        frame = FaceFrame((observation(subject=True), observation(), observation(Rejection.EYES)), 0.0)
        marks = {m.color: m for m in build_marks(frame, charge=1.0)}
        self.assertEqual(marks[SUBJECT_COLOR].line, SUBJECT_LINE_PX)
        self.assertEqual(marks[ACCEPTED_COLOR].line, LINE_PX)
        self.assertEqual(marks[REJECTED_COLOR].line, LINE_PX)

    def test_a_rejected_subject_is_thick_labelled_and_follows_the_charge(self):
        frame = FaceFrame((observation(Rejection.TURNED, subject=True, limit=20.0),), 0.0)
        mark = build_marks(frame)[0]
        self.assertEqual(mark.color, REJECTED_COLOR)
        self.assertEqual(mark.line, SUBJECT_LINE_PX)
        self.assertTrue(mark.label)
        self.assertEqual(build_marks(frame, charge=0.5)[0].color, mix(REJECTED_COLOR, SUBJECT_COLOR, 0.5))

    def test_rejected_first_subject_last(self):
        frame = FaceFrame((observation(subject=True), observation(Rejection.SMALL), observation()), 0.0)
        colors = [m.color for m in build_marks(frame, charge=1.0)]
        self.assertEqual(colors, [REJECTED_COLOR, ACCEPTED_COLOR, SUBJECT_COLOR])

    def test_a_field_per_face(self):
        self.assertEqual(face_fields(FaceFrame((), 0.0)), [])
        frame = FaceFrame((observation(subject=True, bbox=(0.1, 0.2, 0.5, 0.7)),
                           observation(Rejection.SMALL, bbox=(0.5, 0.0, 0.75, 0.25), distance=0.6)), 0.0)
        fields = face_fields(frame)
        self.assertEqual(len(fields), 2)
        # box and distance from good
        for got, want in zip(fields[0] + fields[1], (0.1, 0.2, 0.4, 0.5, 0.0, 0.5, 0.0, 0.25, 0.25, 0.6)):
            self.assertAlmostEqual(got, want)

    def test_the_subject_mark_carries_its_distance(self):
        subject = SubjectMark()
        away = FaceFrame((observation(Rejection.TURNED, subject=True, distance=0.4),), 10.0)
        self.assertEqual(subject.update(FaceFrame((observation(subject=True),), 10.0), 10.0, 0.5).far, 0.0)
        mark = subject.update(away, 10.0, 0.5)
        self.assertEqual((mark.far, mark.subject), (0.4, True))
        self.assertEqual(subject.update(FaceFrame((), 10.1), 10.1, 0.4).far, 1.0)     # the face left: as far as can be
        self.assertEqual(build_marks(away)[0].far, 0.4)

    def test_box_from_bbox(self):
        mark = build_marks(FaceFrame((observation(bbox=(0.1, 0.2, 0.5, 0.7)),), 0.0))[0]
        self.assertAlmostEqual(mark.x, 0.1)
        self.assertAlmostEqual(mark.y, 0.2)
        self.assertAlmostEqual(mark.w, 0.4)
        self.assertAlmostEqual(mark.h, 0.5)


class TestFitBox(unittest.TestCase):
    PAD = (0.01, 0.02)

    def test_a_box_inside_is_unchanged(self):
        self.assertEqual(fit_box((0.1, 0.2, 0.5, 0.7), self.PAD), (0.1, 0.2, 0.5, 0.7))
        self.assertEqual(fit_box((-0.1, 0.2, 0.5, 1.1)), (0.0, 0.2, 0.5, 1.0))     # no padding: the image edge

    def test_an_edge_past_the_padding_is_set_to_it(self):
        self.assertEqual(fit_box((-0.1, -0.3, 0.5, 0.7), self.PAD), (0.01, 0.02, 0.5, 0.7))      # left, top
        self.assertEqual(fit_box((0.1, 0.2, 1.2, 1.1), self.PAD), (0.1, 0.2, 0.99, 0.98))        # right, bottom
        self.assertEqual(fit_box((-0.1, 0.2, 1.2, 0.7), self.PAD), (0.01, 0.2, 0.99, 0.7))       # both sides

    def test_a_box_beyond_the_image_closes_to_nothing(self):
        x0, y0, x1, y1 = fit_box((1.1, 0.2, 1.3, 0.7), self.PAD)
        self.assertEqual((x0, x1), (1.1, 1.1))
        self.assertEqual((y0, y1), (0.2, 0.7))

    def test_every_drawn_box_is_fitted(self):
        face = observation(subject=True, bbox=(-0.1, 0.2, 0.5, 1.1))
        frame = FaceFrame((face,), 10.0)
        mark = build_marks(frame, pad=self.PAD)[0]
        subject = SubjectMark().update(frame, 10.0, 0.5, pad=self.PAD)
        for x, y, w, h in ((mark.x, mark.y, mark.w, mark.h), face_fields(frame, self.PAD)[0][:4],
                           (subject.x, subject.y, subject.w, subject.h)):
            self.assertAlmostEqual(x, 0.01)
            self.assertAlmostEqual(y, 0.2)
            self.assertAlmostEqual(w, 0.49)
            self.assertAlmostEqual(h, 0.78)
        # without a padding the box is cut at the image edge
        self.assertAlmostEqual(build_marks(frame)[0].x, 0.0)
        self.assertAlmostEqual(face_fields(frame)[0][3], 0.8)
        self.assertAlmostEqual(SubjectMark().update(frame, 10.0, 0.5).h, 0.8)


if __name__ == '__main__':
    unittest.main()
