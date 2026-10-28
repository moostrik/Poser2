"""End-to-end: a synthetic room through the real lens, then through the warp, judged as pixels.

Everything else in `modules/oak/tests` checks the mesh point by point. This one does what the
rig does: renders a scene into a raw sensor frame with an INDEPENDENT forward model of the lens
(equidistant, the shared focal and centre, the camera aimed up by the tilt), warps it with the
32 x 32 mesh bilinearly densified — exactly how the Warp node reads a mesh — and measures the
result the way a person with a tape would:

- a vertical pole is one column, anywhere in the frame;
- tape at lens height is one row, the horizon row the window names;
- a ceiling edge lies on `atan(H cos b / D)`, which is where a straight line that is neither
  vertical nor at the horizon belongs in a column-is-azimuth frame (it curves; that is correct);
- a metre is the same number of rows at any height, at a fixed distance (the cylindrical
  property the pose model is given the frame for);
- black is exactly where `frame_coverage` says the sensor did not look.

Set `POSER_TEST_IMAGES` to a directory to have the raw and warped frames written as PNGs.
"""

import math
import os
import unittest
from pathlib import Path

import cv2
import numpy as np

from modules.oak import WARP_MESH, warp_mesh_points, frame_window, frame_coverage, source_lens


W, H = 1280, 800
MODE_W = W
FOV = 127.0
DPP = FOV / W
TILT = 16.0
LENS_FOV = 128.9
LENS_CENTRE = (-10.5, 10.5)
CAM_HEIGHT = 0.5                # lens above the floor, m

FOCAL, LCX, LCY = source_lens((W, H), MODE_W, FOV, LENS_FOV, LENS_CENTRE)
WINDOW = frame_window((W, H), (W, H), MODE_W, FOV, TILT, LENS_FOV, LENS_CENTRE)

BACKGROUND = 40                 # the sensor sees this everywhere; black (0) means "not imaged"
SHIFT = 4                       # cv2 sub-pixel shift bits


def lens_pixel(x: float, y: float, z: float) -> tuple[float, float]:
    """The raw sensor pixel of a world point: x right, y up, z forward, metres, from the lens.
    The camera is aimed up by TILT, so the world ray is rotated down by TILT into the camera's
    frame before the equidistant projection. Written in the plainest form, independently of
    the module."""
    t = math.radians(TILT)
    # World -> camera (aimed up): rotate about the x axis by -tilt.
    yc = y * math.cos(t) - z * math.sin(t)
    zc = y * math.sin(t) + z * math.cos(t)
    xc = x
    norm = math.sqrt(xc * xc + yc * yc + zc * zc)
    phi = math.acos(max(-1.0, min(1.0, zc / norm)))
    psi = math.atan2(-yc, xc)                   # image y is down
    r = FOCAL * phi
    return LCX + r * math.cos(psi), LCY + r * math.sin(psi)


def frame_pixel(x: float, y: float, z: float) -> tuple[float, float]:
    """Where the delivered frame should put that world point: column = bearing, row = tangent
    of elevation below the horizon row."""
    bearing = math.degrees(math.atan2(x, z))
    elevation = math.degrees(math.atan2(y, math.hypot(x, z)))
    return (W - 1) / 2.0 + bearing / DPP, WINDOW.row(elevation)


def draw_polyline(img: np.ndarray, points: list[tuple[float, float]], value: int) -> None:
    pts = np.array([(round(x * (1 << SHIFT)), round(y * (1 << SHIFT))) for x, y in points], dtype=np.int32)
    cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, value, 1, cv2.LINE_AA, shift=SHIFT)


def densify(mesh: np.ndarray, out_w: int, out_h: int) -> tuple[np.ndarray, np.ndarray]:
    """The full-resolution map the Warp node builds from a mesh: bilinear between mesh points,
    which sit at `linspace(0, size - 1, n)` on each axis."""
    n = WARP_MESH
    g = mesh.reshape(n, n, 2)
    u = np.arange(out_w) / (out_w - 1) * (n - 1)
    v = np.arange(out_h) / (out_h - 1) * (n - 1)
    i = np.minimum(u.astype(int), n - 2)
    j = np.minimum(v.astype(int), n - 2)
    fu = (u - i)[None, :, None]
    fv = (v - j)[:, None, None]
    top = g[j][:, i] * (1 - fu) + g[j][:, i + 1] * fu
    bot = g[j + 1][:, i] * (1 - fu) + g[j + 1][:, i + 1] * fu
    full = top * (1 - fv) + bot * fv
    return full[..., 0].astype(np.float32), full[..., 1].astype(np.float32)


class Scene:
    """A room around a lens 0.5 m up: a wall 3 m ahead, a ceiling 2.5 m above the lens, poles,
    tape, ticks marking two one-metre spans. Each primitive is a list of world points, rendered
    both through the lens (a raw frame) and directly into frame coordinates (the expectation).

    Primitives are drawn on separate LAYERS (one raw frame each), so a line being measured is
    never crossed by another one inside the measuring window."""

    WALL = 3.0
    CEILING = 2.5

    def __init__(self) -> None:
        self.raws: dict[str, np.ndarray] = {}
        self.items: dict[str, list[tuple[float, float, float]]] = {}

    def add(self, layer: str, name: str, points: list[tuple[float, float, float]]) -> None:
        self.items[name] = points
        raw = self.raws.setdefault(layer, np.full((H, W), BACKGROUND, dtype=np.uint8))
        draw_polyline(raw, [lens_pixel(*p) for p in points], 255)

    @staticmethod
    def pole(bearing: float, distance: float, top: float = 2.2) -> list[tuple[float, float, float]]:
        b = math.radians(bearing)
        return [(distance * math.sin(b), h - CAM_HEIGHT, distance * math.cos(b)) for h in np.linspace(0.0, top, 40)]

    @staticmethod
    def tick(bearing: float, distance: float, height_above_lens: float) -> list[tuple[float, float, float]]:
        """A short horizontal mark, 16 cm wide, at a height on a vertical through (bearing, distance).
        Its middle point (index 2) is the mark's centre."""
        b = math.radians(bearing)
        xc, zc = distance * math.sin(b), distance * math.cos(b)
        return [(xc + dx, height_above_lens, zc) for dx in np.linspace(-0.08, 0.08, 5)]

    def wall_line(self, height_above_lens: float) -> list[tuple[float, float, float]]:
        return [(self.WALL * math.tan(math.radians(b)), height_above_lens, self.WALL) for b in np.linspace(-62.0, 62.0, 200)]


def brightest_column(img: np.ndarray, row: int, near: float, halfwidth: int = 12) -> float | None:
    lo, hi = max(0, int(near) - halfwidth), min(W, int(near) + halfwidth + 1)
    strip = img[row, lo:hi].astype(float)
    if strip.max() < 100:                       # an anti-aliased 1 px line peaks well above this
        return None
    weights = np.clip(strip - BACKGROUND, 0, None)
    return lo + float((weights * np.arange(len(strip))).sum() / weights.sum())


def brightest_row(img: np.ndarray, column: int, near: float, halfwidth: int = 12) -> float | None:
    lo, hi = max(0, int(near) - halfwidth), min(H, int(near) + halfwidth + 1)
    strip = img[lo:hi, column].astype(float)
    if strip.max() < 100:                       # an anti-aliased 1 px line peaks well above this
        return None
    weights = np.clip(strip - BACKGROUND, 0, None)
    return lo + float((weights * np.arange(len(strip))).sum() / weights.sum())


class LensImageTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        scene = Scene()
        scene.add('tape', 'tape', scene.wall_line(0.0))
        scene.add('ceiling', 'ceiling', scene.wall_line(Scene.CEILING))
        for bearing in (10.0, 45.0, 60.0):
            scene.add('poles', f'pole{int(bearing)}', Scene.pole(bearing, 2.5))
        scene.add('poles', 'pole-30', Scene.pole(-30.0, 1.6))
        # Two one-metre spans on one vertical, 2.5 m out at 30 deg, marked by ticks: one at eye
        # level, one overhead.
        for name, (h0, h1) in (('low', (0.3, 1.3)), ('high', (1.0, 2.0))):
            scene.add('ticks', f'tick_{name}_a', Scene.tick(30.0, 2.5, h0))
            scene.add('ticks', f'tick_{name}_b', Scene.tick(30.0, 2.5, h1))
        cls.scene = scene

        mesh = np.array(warp_mesh_points((W, H), (W, H), MODE_W, FOV, TILT, False, False,
                                         WARP_MESH, WARP_MESH, LENS_FOV, LENS_CENTRE))
        map_x, map_y = densify(mesh, W, H)
        cls.warped = {layer: cv2.remap(raw, map_x, map_y, cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                      for layer, raw in scene.raws.items()}
        cls.coverage = frame_coverage((W, H), (W, H), MODE_W, FOV, TILT, False, False, LENS_FOV, LENS_CENTRE)

        out_dir = os.environ.get('POSER_TEST_IMAGES')
        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(out_dir) / 'lens_raw.png'), np.max(list(scene.raws.values()), axis=0))
            cv2.imwrite(str(Path(out_dir) / 'lens_warped.png'), np.max(list(cls.warped.values()), axis=0))

    def _frame_points(self, name: str) -> list[tuple[float, float]]:
        return [frame_pixel(*p) for p in self.scene.items[name]]

    def test_a_vertical_pole_is_one_column(self) -> None:
        for name in ('pole10', 'pole45', 'pole60', 'pole-30'):
            with self.subTest(pole=name):
                expected_x = self._frame_points(name)[0][0]
                columns = []
                for x, y in self._frame_points(name)[2:-2]:       # not the line's own ends
                    row = int(round(y))
                    if not (2 <= row <= H - 3):
                        continue
                    found = brightest_column(self.warped['poles'], row, x)
                    self.assertIsNotNone(found, f'no pole pixel on row {row}')
                    columns.append(found)
                self.assertGreater(len(columns), 10)
                self.assertLess(max(columns) - min(columns), 1.0)
                self.assertAlmostEqual(float(np.mean(columns)), expected_x, delta=0.75)

    def test_tape_at_lens_height_is_the_horizon_row(self) -> None:
        rows = []
        for column in range(40, W - 40, 40):
            found = brightest_row(self.warped['tape'], column, WINDOW.horizon_px)
            self.assertIsNotNone(found, f'no tape on column {column}')
            rows.append(found)
        self.assertLess(max(rows) - min(rows), 1.0)
        self.assertAlmostEqual(float(np.mean(rows)), WINDOW.horizon_px, delta=0.75)

    def test_the_ceiling_curves_as_the_geometry_says(self) -> None:
        # A straight edge H above the lens on a wall D ahead sits at atan(H cos b / D): highest
        # straight ahead, lower toward the sides. Curved, and exactly this curve.
        for column in range(60, W - 60, 60):
            bearing = (column - (W - 1) / 2.0) * DPP
            elevation = math.degrees(math.atan(Scene.CEILING * math.cos(math.radians(bearing)) / Scene.WALL))
            expected = WINDOW.row(elevation)
            if not (2 <= expected <= H - 3):
                continue
            with self.subTest(column=column):
                found = brightest_row(self.warped['ceiling'], column, expected)
                self.assertIsNotNone(found)
                self.assertAlmostEqual(found, expected, delta=1.0)
        # And it is genuinely curved: the centre sits well above the sides.
        centre = WINDOW.row(math.degrees(math.atan(Scene.CEILING / Scene.WALL)))
        side = WINDOW.row(math.degrees(math.atan(Scene.CEILING * math.cos(math.radians(55.0)) / Scene.WALL)))
        self.assertGreater(side - centre, 100.0)

    def test_a_metre_is_the_same_height_anywhere_in_the_frame(self) -> None:
        # The cylindrical property: at a fixed horizontal distance a vertical metre spans
        # FOCAL_out / distance rows whether it stands at eye level or overhead.
        spans = {}
        for name in ('low', 'high'):
            rows = []
            for end in ('a', 'b'):
                x, y = self._frame_points(f'tick_{name}_{end}')[2]        # the tick's centre
                found = brightest_row(self.warped['ticks'], int(round(x)), y, halfwidth=6)
                self.assertIsNotNone(found, f'no tick {name} {end}')
                rows.append(found)
            spans[name] = abs(rows[0] - rows[1])
        expected = WINDOW.focal * 1.0 / 2.5
        self.assertAlmostEqual(spans['low'], expected, delta=2.0)
        self.assertAlmostEqual(spans['high'], expected, delta=2.0)

    def test_black_is_exactly_where_the_sensor_did_not_look(self) -> None:
        imaged = self.warped['tape'] > 0
        for column in range(0, W, 16):
            with self.subTest(column=column):
                rows = np.flatnonzero(imaged[:, column])
                first, last = self.coverage[column]
                if first < 0:
                    self.assertEqual(len(rows), 0)
                    continue
                self.assertLessEqual(abs(int(rows[0]) - int(first)), 2)
                self.assertLessEqual(abs(int(rows[-1]) - int(last)), 2)


if __name__ == "__main__":
    unittest.main()
