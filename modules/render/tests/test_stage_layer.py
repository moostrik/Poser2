"""A stage's reset button asks for a reset; the stage's next update takes it, once, before it runs."""

import unittest
from dataclasses import dataclass
from typing import Annotated

from modules.gl import Texture
from modules.flow import HasPorts, Port, Ports, TextureKind, ports_of, prefixed
from modules.render.layers import StageLayer, StageLayerSettings, StageOutput


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

@dataclass
class _Input:
    image: Annotated[Texture, TextureKind.IMAGE]
    dt: float
    mask: Annotated[Texture | None, TextureKind.SCALAR] = None


@dataclass
class _KindOutput(StageOutput):
    velocity: Annotated[Texture, TextureKind.VELOCITY]
    impulse: Annotated[Texture | None, TextureKind.VELOCITY] = None


def _allocated() -> Texture:
    texture = Texture()
    texture.allocated = True
    return texture


class _PartStage(StageLayer[_Input, _KindOutput]):
    """A stage with two parts, one of which owns a texture that is not allocated."""
    def __init__(self) -> None:
        super().__init__(StageLayerSettings())
        self.velocity, self.trail, self.idle = _allocated(), _allocated(), Texture()

    @property
    def output(self) -> _KindOutput:
        return _KindOutput(self.velocity)

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None: ...

    def reset(self) -> None: ...

    def _update(self) -> None: ...

    def _components(self) -> dict[str, HasPorts]:
        return {'trail': Ports({'smooth': Port(self.trail, TextureKind.VELOCITY), 'idle': Port(self.idle, TextureKind.SCALAR)}),
                '': Ports({'color': Port(self.trail, TextureKind.IMAGE)})}


class PortsTest(unittest.TestCase):
    """A field that declares its kind is a port; one left None is none; a stage lists its output,
    its parts and its input with the data running bottom to top, and leaves out what is not allocated."""

    def test_a_dataclass_gives_its_set_fields_with_their_kinds(self) -> None:
        image = Texture()
        ports = ports_of(_Input(image, 0.016))
        self.assertEqual(list(ports), ['image'])
        self.assertIs(ports['image'].texture, image)
        self.assertEqual(ports['image'].kind, TextureKind.IMAGE)

    def test_a_stage_lists_output_parts_input_top_down_without_unallocated(self) -> None:
        stage = _PartStage()
        stage.set_input(_Input(_allocated(), 0.016))
        self.assertEqual(list(stage.ports), ['output.velocity', 'color', 'trail.smooth', 'input.image'])
        self.assertEqual(stage.ports['output.velocity'].kind, TextureKind.VELOCITY)

    def test_prefixed_names_a_part_and_leaves_no_prefix_alone(self) -> None:
        port = Port(Texture(), TextureKind.SCALAR)
        self.assertEqual(list(prefixed('z', {'track': port})), ['z.track'])
        self.assertEqual(list(prefixed('', {'track': port})), ['track'])


if __name__ == '__main__':
    unittest.main()
