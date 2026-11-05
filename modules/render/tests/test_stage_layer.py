"""A stage's reset button asks for a reset; the stage's next update takes it, once, before it runs."""

import unittest
from dataclasses import dataclass

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

    def test_an_output_takes_its_ports_by_keyword_after_its_own_fields(self) -> None:
        output = _Output(3, textures={})
        self.assertEqual((output.value, output.textures), (3, {}))


if __name__ == '__main__':
    unittest.main()
