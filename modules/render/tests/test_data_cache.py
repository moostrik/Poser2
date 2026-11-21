"""The cache a layer keeps of the data it last drew: changed, idle and lost by identity."""

import unittest

from modules.render import DataCache


class DataCacheTest(unittest.TestCase):
    def test_starts_empty(self) -> None:
        cache: DataCache[object] = DataCache()
        self.assertTrue(cache.empty)
        self.assertFalse(cache.has_data)
        self.assertIsNone(cache.data)
        self.assertFalse(cache.changed)
        self.assertFalse(cache.lost)

    def test_new_data_is_a_change(self) -> None:
        cache: DataCache[object] = DataCache()
        item = object()
        cache.set(item)
        self.assertTrue(cache.changed)
        self.assertFalse(cache.idle)
        self.assertFalse(cache.lost)
        self.assertIs(cache.data, item)
        self.assertTrue(cache.has_data)

    def test_the_same_item_again_is_idle(self) -> None:
        cache: DataCache[object] = DataCache()
        item = object()
        cache.set(item)
        cache.set(item)
        self.assertTrue(cache.idle)
        self.assertFalse(cache.changed)

    def test_an_equal_but_other_item_is_a_change(self) -> None:
        cache: DataCache[list[int]] = DataCache()
        cache.set([1])
        cache.set([1])
        self.assertTrue(cache.changed)

    def test_none_after_data_is_lost_once(self) -> None:
        cache: DataCache[object] = DataCache()
        cache.set(object())
        cache.set(None)
        self.assertTrue(cache.lost)
        self.assertTrue(cache.changed)
        self.assertTrue(cache.empty)
        cache.set(None)
        self.assertFalse(cache.lost)
        self.assertTrue(cache.idle)

    def test_none_while_empty_is_idle(self) -> None:
        cache: DataCache[object] = DataCache()
        cache.set(None)
        self.assertTrue(cache.idle)
        self.assertFalse(cache.lost)


if __name__ == '__main__':
    unittest.main()
