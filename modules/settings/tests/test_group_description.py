"""A group's description: one line of panel text, reached through its parent's class, a subclass's included."""

import unittest

from modules.settings import BaseSettings, Field, Group, mirror


class _Child(BaseSettings):
    value: Field[float] = Field(0.0)


class _Base(BaseSettings):
    described = Group(_Child, description="What the child is")
    plain = Group(_Child)


class _Sub(_Base):
    pass


class GroupDescriptionTest(unittest.TestCase):
    def test_the_description_is_on_the_descriptor(self) -> None:
        group = getattr(_Sub, 'described')
        self.assertIsInstance(group, Group)
        self.assertEqual(group.description, "What the child is")

    def test_none_given_is_empty(self) -> None:
        self.assertEqual(getattr(_Base, 'plain').description, "")
        self.assertEqual(getattr(_Base, 'plain').tint, "")

    def test_a_tint_is_on_the_descriptor(self) -> None:
        class _Tinted(BaseSettings):
            child = Group(_Child, tint="#4d5a4f")
        self.assertEqual(getattr(_Tinted, 'child').tint, "#4d5a4f")

    def test_the_mirror_keeps_the_hints(self) -> None:
        class _Tinted(BaseSettings):
            child = Group(_Child, description="What the child is", tint="#4d5a4f")
        settings = _Tinted()
        settings.initialize()
        mirrored = mirror.build_mirror(mirror.describe(settings))
        group = getattr(type(mirrored), 'child')
        self.assertEqual((group.description, group.tint), ("What the child is", "#4d5a4f"))

    def test_the_settings_file_is_unchanged(self) -> None:
        settings = _Base()
        settings.initialize()
        self.assertEqual(settings.to_dict(), {'described': {'value': 0.0}, 'plain': {'value': 0.0}})
