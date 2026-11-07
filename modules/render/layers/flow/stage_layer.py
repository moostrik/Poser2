"""A stage of a flow system: what the motion, the bridge and the fluid layer are.

A stage takes an input the app builds each tick (`set_input`), hands out an output (`output`), and
starts from rest on `reset`, which its panel's reset button asks for. The next stage reads the
output's fields by name. The monitors read the stage's `ports`, one list with the data running
bottom to top: the output's fields as `output.<field>` on top, each component's textures under its
settings group's name (`trail.smooth`, `xy.solve`), the input's fields as `input.<field>` at the bottom. Each texture is
written once: a dataclass field declares its kind, a component names its own in its `ports`.
Inputs share no base: each stage reads its own fields, and nothing takes any stage's input."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import Event

from modules.flow import HasPorts, Port, ports_of, prefixed
from modules.settings import Field, BaseSettings, Widget
from ..layer_base import LayerBase


@dataclass
class StageOutput:
    """The base of every stage output: its fields that declare a kind are its ports."""

    @property
    def ports(self) -> dict[str, Port]:
        """The fields that declare a kind, by name."""
        return ports_of(self)


class StageLayerSettings(BaseSettings):
    """What every stage layer's settings hold, first on its panel: the reset and the stage's working
    size, which each stage's settings declare again with its own meaning and default."""
    reset: Field[bool] =    Field(False, widget=Widget.button, description="Start the stage from rest")
    width: Field[int] =     Field(512, min=32, max=4096, step=32, description="Working width")
    height: Field[int] =    Field(288, min=32, max=4096, step=32, description="Working height")


class Stage3DLayerSettings(StageLayerSettings):
    """A stage whose working size has a depth: its z layers right after its width and height."""
    layers: Field[int] =    Field(16, min=1, max=256, description="Working layers in z")


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
        """What the monitors show of the stage, the data running bottom to top: `output.*` on top,
        each component's under its name from the last the stage runs to the first, `input.*` at the
        bottom. A texture not allocated is left out, so a part the stage does not run gives none."""
        inputs: dict[str, Port] = ports_of(self._input) if self._input is not None else {}
        parts: dict[str, Port] = {}
        for name, component in self._components().items():
            parts.update(prefixed(name, component.ports))
        ports: dict[str, Port] = {**prefixed('input', inputs), **parts, **prefixed('output', self.output.ports)}
        return {name: port for name, port in reversed(ports.items()) if port.texture.allocated}

    def _components(self) -> dict[str, HasPorts]:
        """The parts that own textures, by their settings group's name ('' for none), in the order
        the stage runs them. A stage with parts lists them here once."""
        return {}

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
