"""No flow layer's instance attribute shadows one of its methods: an attribute a base class sets in its
constructor and a method a subclass defines under the same name leave the method uncallable at runtime."""

import inspect
import unittest

from modules.render.layers import Bridge2DLayer, Bridge2DLayerSettings, Bridge3DLayer, Bridge3DLayerSettings, \
                                  Fluid2DLayer, Fluid2DLayerSettings, Fluid3DLayer, Fluid3DLayerSettings, \
                                  Motion2DLayer, Motion2DLayerSettings, Motion3DLayer, Motion3DLayerSettings, \
                                  MonitorLayer, MonitorLayerSettings

def layers() -> list[object]:
    """One of each flow layer; their constructors make no GL calls."""
    return [Motion2DLayer(Motion2DLayerSettings()), Motion3DLayer(0, Motion3DLayerSettings()),
            Bridge2DLayer(Bridge2DLayerSettings()), Bridge3DLayer(Bridge3DLayerSettings()),
            Fluid2DLayer(Fluid2DLayerSettings()), Fluid3DLayer(Fluid3DLayerSettings()),
            MonitorLayer(MonitorLayerSettings())]


class LayerNamesTest(unittest.TestCase):
    def test_no_attribute_shadows_a_method(self) -> None:
        for layer in layers():
            with self.subTest(layer=type(layer).__name__):
                methods: set[str] = {name for name, member in inspect.getmembers(type(layer)) if callable(member)}
                self.assertEqual(sorted(methods & set(vars(layer))), [])


if __name__ == '__main__':
    unittest.main()
