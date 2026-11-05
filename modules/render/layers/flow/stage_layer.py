"""A stage of a flow system: what the motion, the bridge and the fluid layer are.

A stage takes an input the app builds each tick (`set_input`), hands out an output (`output`), and
starts from rest on `reset`, which its panel's reset button asks for. The next stage reads the
output's fields by name. The monitors read the stage's `ports`: every input and output field that
declares its kind, written once on the field, and the `diagnostics` a stage adds of its insides.
Inputs share no base: each stage reads its own fields, and nothing takes any stage's input."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from threading import Event

from modules.settings import Field, BaseSettings, Widget
from ..layer_base import LayerBase
from .port import Port, ports_of


@dataclass
class StageOutput:
    """What every stage output carries beside its fields: `diagnostics`, the stage's insides a
    monitor can show, which no stage downstream reads."""
    diagnostics: dict[str, Port] = field(default_factory=dict, kw_only=True)

    @property
    def ports(self) -> dict[str, Port]:
        """The fields that declare a kind, then the diagnostics."""
        return {**ports_of(self), **self.diagnostics}


class StageLayerSettings(BaseSettings):
    """What every stage layer's settings hold, first on its panel: the reset and the stage's working
    size, which each stage's settings declare again with its own meaning and default."""
    reset: Field[bool] =    Field(False, widget=Widget.button, description="Start the stage from rest")
    width: Field[int] =     Field(512, min=32, max=4096, step=32, description="Working width")
    height: Field[int] =    Field(288, min=32, max=4096, step=32, description="Working height")


class StageLayer[I, O: StageOutput](LayerBase, ABC):
    """A stage's layer: `set_input` before the render loop, `output` its output, `reset` from rest.
    `update` resets first when the panel asked for it, on the render thread, then runs the stage
    (`_update`). A subclass that overrides `deallocate` calls this one, which unbinds the button."""

    def __init__(self, settings: StageLayerSettings) -> None:
        self.settings: StageLayerSettings = settings
        self._input: I | None = None
        self._reset_asked: Event = Event()      # set by the panel's thread, taken by the render thread
        settings.bind(StageLayerSettings.reset, self._ask_reset)

    def set_input(self, input: I) -> None:
        """What the stage takes at its next update."""
        self._input = input

    @property
    @abstractmethod
    def output(self) -> O:
        """The stage's output, its textures allocated once; what the next stage's input is built from."""
        ...

    @property
    def ports(self) -> dict[str, Port]:
        """What the monitors show of the stage: its output's ports, then its input's, named `input_<field>`."""
        inputs: dict[str, Port] = ports_of(self._input, 'input_') if self._input is not None else {}
        return {**self.output.ports, **inputs}

    @abstractmethod
    def reset(self) -> None:
        """Start the stage from rest. Render thread only."""

    def deallocate(self) -> None:
        self.settings.unbind(StageLayerSettings.reset, self._ask_reset)

    def update(self) -> None:
        if self._reset_asked.is_set():
            self._reset_asked.clear()
            self.reset()
        self._update()

    @abstractmethod
    def _update(self) -> None:
        """Run the stage on its input for this tick."""

    def _ask_reset(self, _: object = None) -> None:
        self._reset_asked.set()
