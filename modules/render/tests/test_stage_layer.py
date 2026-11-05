"""A stage's reset button asks for a reset; the stage's next update takes it, once, before it runs."""

import unittest
from dataclasses import dataclass
from typing import Annotated

from modules.gl import Texture
from modules.render.layers import Port, StageLayer, StageLayerSettings, StageOutput, TextureKind, ports_of


@dataclass
class _Output(StageOutput):
    value: int = 0


class _Stage(StageLayer[int, _Output]):
    def __init__(self, settings: StageLayerSettings) -> None:
        super().__init__(settings)
        self.calls: list[str] = []

    @property
    def output(self) -> _Output:
        return _Output(1)

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None: ...

    def reset(self) -> None:
        self.calls.append('reset')

    def _update(self) -> None:
        self.calls.append('update')


class StageLayerTest(unittest.TestCase):
    def test_the_button_resets_once_at_the_next_update_before_the_stage_runs(self) -> None:
        settings = StageLayerSettings()
        stage = _Stage(settings)
        StageLayerSettings.reset.fire(settings)
        self.assertEqual(stage.calls, [])                       # nothing on the panel's thread
        stage.update()
        stage.update()
        self.assertEqual(stage.calls, ['reset', 'update', 'update'])

    def test_after_deallocate_the_button_asks_nothing(self) -> None:
        settings = StageLayerSettings()
        stage = _Stage(settings)
        stage.deallocate()
        StageLayerSettings.reset.fire(settings)
        stage.update()
        self.assertEqual(stage.calls, ['update'])

    def test_an_output_takes_its_diagnostics_by_keyword_after_its_own_fields(self) -> None:
        output = _Output(3, diagnostics={})
        self.assertEqual((output.value, output.diagnostics), (3, {}))


@dataclass
class _Input:
    image: Annotated[Texture, TextureKind.IMAGE]
    dt: float
    mask: Annotated[Texture | None, TextureKind.SCALAR] = None


@dataclass
class _KindOutput(StageOutput):
    velocity: Annotated[Texture, TextureKind.VELOCITY]
    onset: Annotated[Texture | None, TextureKind.VELOCITY] = None


class PortsTest(unittest.TestCase):
    """A field that declares its kind is a port; one left None is none; a diagnostic comes after."""

    def test_an_input_gives_its_set_fields_with_their_kinds(self) -> None:
        image = Texture()
        ports = ports_of(_Input(image, 0.016), 'input_')
        self.assertEqual(list(ports), ['input_image'])
        self.assertIs(ports['input_image'].texture, image)
        self.assertEqual(ports['input_image'].kind, TextureKind.IMAGE)

    def test_an_output_gives_its_fields_then_its_diagnostics(self) -> None:
        velocity, inner = Texture(), Texture()
        output = _KindOutput(velocity, diagnostics={'inner': Port(inner, TextureKind.SCALAR)})
        self.assertEqual(list(output.ports), ['velocity', 'inner'])
        self.assertEqual(output.ports['velocity'].kind, TextureKind.VELOCITY)


if __name__ == '__main__':
    unittest.main()
