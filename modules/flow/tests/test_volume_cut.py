"""A volume port says it is one; its owner's refresh runs once per change and only when asked; a cut
has the box's proportions; a projection takes the max only for an image; the histogram reads a
volume through all its layers."""

import unittest

from modules.gl import Texture, Texture3D
from modules.flow import Cut, OnDemand, Port, TextureKind, cut_size, histogram_volume_grid, projects_max


class VolumePortTest(unittest.TestCase):
    def test_a_3d_texture_is_a_volume_and_a_2d_one_is_not(self) -> None:
        self.assertTrue(Port(Texture3D(), TextureKind.SCALAR).volume)
        self.assertFalse(Port(Texture(), TextureKind.SCALAR).volume)


class OnDemandTest(unittest.TestCase):
    def test_it_runs_once_per_mark_however_many_ask(self) -> None:
        runs: list[int] = []
        refresh = OnDemand(lambda: runs.append(1))
        refresh()
        refresh()
        self.assertEqual(len(runs), 1)
        refresh.mark()
        refresh()
        refresh()
        self.assertEqual(len(runs), 2)

    def test_it_never_runs_when_nobody_asks(self) -> None:
        runs: list[int] = []
        refresh = OnDemand(lambda: runs.append(1))
        for _ in range(3):
            refresh.mark()
        self.assertEqual(runs, [])


class CutTest(unittest.TestCase):
    def test_a_slice_and_the_front_are_the_face(self) -> None:
        self.assertEqual(cut_size(Cut.SLICE, 448, 800, 0.14), (448, 800))
        self.assertEqual(cut_size(Cut.FRONT, 448, 800, 0.14), (448, 800))

    def test_the_top_and_the_side_are_as_deep_as_the_box(self) -> None:
        self.assertEqual(cut_size(Cut.TOP, 448, 800, 0.25), (448, 112))
        self.assertEqual(cut_size(Cut.SIDE, 448, 800, 0.25), (112, 800))

    def test_the_cube_and_the_fog_are_3d_the_rest_flat(self) -> None:
        self.assertEqual([cut for cut in Cut if not cut.flat], [Cut.CUBE, Cut.FOG])

    def test_only_an_image_projects_its_max(self) -> None:
        self.assertTrue(projects_max(TextureKind.IMAGE))
        self.assertFalse(projects_max(TextureKind.VELOCITY))
        self.assertFalse(projects_max(TextureKind.SCALAR))

    def test_the_histogram_reads_every_layer_of_a_volume(self) -> None:
        self.assertEqual(histogram_volume_grid(448, 800, 16), (128, 128, 16))
        self.assertEqual(histogram_volume_grid(64, 32, 4), (64, 32, 4))


if __name__ == '__main__':
    unittest.main()
