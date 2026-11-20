"""Tests for a Group's defaults: a child's starting values in one place, the class defaults elsewhere."""

import unittest

from modules.settings import BaseSettings, Field, Group


class _Child(BaseSettings):
    enabled: Field[bool] = Field(True)
    rate: Field[float] = Field(1.0)


class _Parent(BaseSettings):
    rate: Field[float] = Field(5.0)
    first: Group[_Child] = Group(_Child)
    second: Group[_Child] = Group(_Child, defaults={'enabled': False})
    shared: Group[_Child] = Group(_Child, share=[rate], defaults={'rate': 2.0})


class TestGroupDefaults(unittest.TestCase):
    def test_a_group_starts_its_child_at_its_defaults(self):
        parent = _Parent()
        self.assertTrue(parent.first.enabled)
        self.assertFalse(parent.second.enabled)

    def test_a_shared_field_takes_the_parents_value(self):
        self.assertEqual(_Parent().shared.rate, 5.0)

    def test_a_settings_file_value_still_wins(self):
        parent = _Parent()
        parent.update_from_dict({'second': {'enabled': True}})
        self.assertTrue(parent.second.enabled)

    def test_an_unknown_field_is_an_error(self):
        class _Wrong(BaseSettings):
            child: Group[_Child] = Group(_Child, defaults={'missing': 1})
        with self.assertRaises(TypeError):
            _Wrong()


if __name__ == '__main__':
    unittest.main()
