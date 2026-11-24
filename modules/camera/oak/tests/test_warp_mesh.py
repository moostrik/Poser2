"""`build_warp_mesh` hands each OAK-D the one correction it uses, as depthai points: the keystone alone gives the
pre-step-3 builders' mesh (`geometry/tests/test_keystone.py`), a tilt the dense warp mesh. On a depthai 2 install
(`.venv`)."""

import unittest

import numpy as np

from modules.camera.geometry import WARP_MESH, MountConfig
from modules.camera.geometry.tests.test_keystone import reference_square, reference_wide, as_float32, assert_same
from modules.camera.oak.warp_mesh import build_warp_mesh


def as_tuples(points) -> np.ndarray:
    return np.array([(p.x, p.y) for p in points], dtype=np.float32)


class BuildWarpMeshSelectsTest(unittest.TestCase):
    """`build_warp_mesh` hands each camera the one correction it uses."""

    def _mount(self, tilt: float, keystone: float, flip_h: bool = True) -> MountConfig:
        return MountConfig(flip_h=flip_h, flip_v=False, tilt=tilt, keystone=keystone, fov_h=127.0)

    def test_keystone_alone_uses_the_old_builder_square(self) -> None:
        # deep_flow: 1280x720 mode, square 720 output, keystone -0.22, flip_h.
        mesh, mw, mh = build_warp_mesh((1280, 720), (720, 720), 1280, self._mount(0.0, -0.22))
        self.assertEqual((mw, mh), (2, 64))
        want = reference_square(1280, 720, 720, 720 * 0.5 * -0.22, True, False, 2, 64)
        assert_same(self, as_tuples(mesh), as_float32(want))

    def test_keystone_alone_uses_the_old_builder_wide(self) -> None:
        mesh, mw, mh = build_warp_mesh((1280, 800), (1280, 800), 1280, self._mount(0.0, -0.22))
        self.assertEqual((mw, mh), (2, 64))
        want = reference_wide(1280, 800, 1280 * 0.5 * -0.22, True, False, 2, 64)
        assert_same(self, as_tuples(mesh), as_float32(want))

    def test_tilt_alone_uses_the_dense_tilt_mesh(self) -> None:
        mesh, mw, mh = build_warp_mesh((1280, 800), (1280, 800), 1280, self._mount(15.0, 0.0))
        self.assertEqual((mw, mh), (WARP_MESH, WARP_MESH))
        self.assertEqual(len(mesh), WARP_MESH * WARP_MESH)

    def test_both_zero_is_the_reprojection_not_identity(self) -> None:
        # No keystone and no tilt still goes through the tilt path, which reprojects the
        # equidistant frame to the cylindrical contract. The mesh is mirror-symmetric about the
        # center column (an even grid has no point ON it, so the two middle columns average to
        # it); the corners are not the identity, because a fisheye's corners bow.
        mesh, mw, mh = build_warp_mesh((1280, 800), (1280, 800), 1280, self._mount(0.0, 0.0, flip_h=False))
        pts = as_tuples(mesh).reshape(mh, mw, 2)
        cx = 1279.0 / 2.0
        w = mw // 2
        np.testing.assert_allclose(pts[:, w - 1:w + 1, 0].mean(axis=1), cx, atol=1e-2)
        np.testing.assert_allclose(pts[:, :, 0], 2.0 * cx - pts[:, ::-1, 0], atol=1e-2)
        top_left = pts[0, 0]
        self.assertGreater(float(np.hypot(top_left[0] - 0.0, top_left[1] - 0.0)), 1.0)

    def test_both_set_warns_and_takes_tilt(self) -> None:
        with self.assertLogs("modules.camera.oak.warp_mesh", level="WARNING") as log:
            mesh, mw, mh = build_warp_mesh((1280, 800), (1280, 800), 1280, self._mount(10.0, -0.22))
        self.assertEqual((mw, mh), (WARP_MESH, WARP_MESH))
        self.assertTrue(any("exclusive" in line for line in log.output))


if __name__ == "__main__":
    unittest.main()
