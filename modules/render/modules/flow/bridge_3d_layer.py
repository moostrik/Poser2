"""The 3D bridge layer: `Bridge3D`, the velocity lowpass and bandpass and the outputs' mixes, on the motion's
velocity (xyz)."""

# Local application imports
from modules.flow import Bridge3D, Bridge3DSettings, HasPorts
from .bridge_layer import BridgeLayer, BridgeLayerSettings, BridgeInput, BridgeOutput


class Bridge3DLayerSettings(Bridge3DSettings, BridgeLayerSettings):
    """Configuration for Bridge3DLayer: the stage's settings and its bridge's own, one panel, the stage's
    first."""


class Bridge3DLayer(BridgeLayer):
    """The 3D bridge layer: hands `Bridge3D` the input and hands out its outputs; the bridge is in
    modules/flow/BRIDGE_3D.md."""

    def __init__(self, settings: Bridge3DLayerSettings) -> None:
        super().__init__(settings)
        self.settings: Bridge3DLayerSettings = settings
        self._bridge_3d: Bridge3D = Bridge3D(settings)

    @property
    def output(self) -> BridgeOutput:
        bridge: Bridge3D = self._bridge_3d
        return BridgeOutput(
            bridge.velocity, bridge.density, bridge.temperature, bridge.acceleration,
            bridge.velocity_z_position, bridge.density_z_position, bridge.temperature_z_position,
            bridge.acceleration_z_position, bridge.expansion, bridge.expansion_z_position)

    def _components(self) -> dict[str, HasPorts]:
        return {'': self._bridge_3d}

    # ========== Lifecycle ==========

    def _allocate_stages(self, width: int, height: int, output_width: int, output_height: int) -> None:
        self._bridge_3d.allocate(width, height, output_width, output_height)

    def _deallocate_stages(self) -> None:
        self._bridge_3d.deallocate()

    def _reset_stages(self) -> None:
        self._bridge_3d.reset()

    # ========== The bridge ==========

    def _bridge(self, input: BridgeInput) -> None:
        self._bridge_3d.update(input.velocity, input.dt, input.z_position, input.mask, input.image, input.color,
                               input.same_tick)
