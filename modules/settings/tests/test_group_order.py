"""A group declared `last` comes after the other groups, a subclass's included."""

import unittest

from modules.settings import BaseSettings, Field, Group


class _Child(BaseSettings):
    value: Field[float] = Field(0.0)


class _Base(BaseSettings):
    first = Group(_Child)
    end = Group(_Child, last=True)


class _Sub(_Base):
    own = Group(_Child)


class GroupOrderTest(unittest.TestCase):
    def test_last_after_a_subclass_groups(self) -> None:
        self.assertEqual(list(_Sub().children), ['first', 'own', 'end'])

    def test_preset_order_follows(self) -> None:
        settings = _Sub()
        settings.initialize()
        self.assertEqual([key for key in settings.to_dict() if key in settings.children], ['first', 'own', 'end'])


if __name__ == '__main__':
    unittest.main()
