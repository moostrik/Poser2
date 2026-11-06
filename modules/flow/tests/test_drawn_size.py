"""A view renders at the largest viewport of the tick before, grows at once to a larger one, and
the summaries read a texture at its own grid up to their caps."""

import unittest

from modules.flow import DrawnSize, histogram_grid, summary_grid


class DrawnSizeTest(unittest.TestCase):
    def test_the_first_draw_renders_at_its_viewport(self) -> None:
        size = DrawnSize()
        size.tick()
        self.assertEqual(size.see(800, 1000), (800, 1000))

    def test_a_larger_draw_in_the_same_tick_grows_it_at_once(self) -> None:
        size = DrawnSize()
        size.tick()
        size.see(400, 500)
        self.assertEqual(size.see(800, 1000), (800, 1000))

    def test_a_smaller_layout_takes_over_one_tick_later(self) -> None:
        size = DrawnSize()
        size.tick()
        size.see(800, 1000)
        size.tick()
        self.assertEqual(size.see(400, 500), (800, 1000))      # the tick before was larger
        size.tick()
        self.assertEqual(size.see(400, 500), (400, 500))

    def test_two_draws_a_tick_keep_the_larger(self) -> None:
        size = DrawnSize()
        for _ in range(3):
            size.tick()
            size.see(400, 500)
            size.see(800, 1000)
        size.tick()
        self.assertEqual(size.see(400, 500), (800, 1000))


class GridTest(unittest.TestCase):
    def test_a_small_texture_is_read_whole(self) -> None:
        self.assertEqual(histogram_grid(512, 288), (512, 288))
        self.assertEqual(summary_grid(512, 288), (512, 288))

    def test_a_large_texture_is_read_on_the_capped_grid(self) -> None:
        self.assertEqual(histogram_grid(2160, 3840), (512, 512))
        self.assertEqual(summary_grid(2160, 3840), (512, 512))


if __name__ == '__main__':
    unittest.main()
