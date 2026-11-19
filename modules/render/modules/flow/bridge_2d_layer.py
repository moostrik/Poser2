"""The 2D bridge layer: `Bridge2D`, the velocity lowpass and bandpass and the outputs' mixes, on the motion's velocity
(xy)."""

# Local application imports
from modules.flow import Bridge2D, BridgeSettings, HasPorts
from .bridge_layer import BridgeLayer, BridgeLayerSettings, BridgeInput, BridgeOutput


class Bridge2DLayerSettings(BridgeSettings, BridgeLayerSettings):
    """Configuration for Bridge2DLayer: the stage's settings and its bridge's own, one panel, the stage's first."""


class Bridge2DLayer(BridgeLayer):
    """The 2D bridge layer: hands `Bridge2D` the input and hands out its outputs; the bridge is in
    modules/flow/BRIDGE_2D.md."""

    def __init__(self, settings: Bridge2DLayerSettings) -> None:
        super().__init__(settings)
        self.settings: Bridge2DLayerSettings = settings
        self._bridge_2d: Bridge2D = Bridge2D(settings)

    @property
    def output(self) -> BridgeOutput:
        bridge: Bridge2D = self._bridge_2d
        return BridgeOutput(bridge.velocity, bridge.density, bridge.color, bridge.temperature, bridge.acceleration,
                            expansion=bridge.expansion)

    def _components(self) -> dict[str, HasPorts]:
        return {'': self._bridge_2d}

    # ========== Lifecycle ==========

    def _allocate_stages(self, width: int, height: int, output_width: int, output_height: int) -> None:
        self._bridge_2d.allocate(width, height, output_width, output_height)

    def _deallocate_stages(self) -> None:
        self._bridge_2d.deallocate()

    def _reset_stages(self) -> None:
        self._bridge_2d.reset()

    # ========== The bridge ==========

    def _bridge(self, input: BridgeInput) -> None:
        self._bridge_2d.update(input.velocity, input.dt, input.mask, input.image, input.same_tick)
