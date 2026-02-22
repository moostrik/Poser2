"""Child — descriptor for nesting a BaseSettings inside another BaseSettings."""

from __future__ import annotations

import logging
from typing import Generic, TypeVar, overload, Any

logger = logging.getLogger(__name__)

C = TypeVar("C")


class Child(Generic[C]):
    """Descriptor that nests one BaseSettings inside another.

    Declare as a class attribute alongside Setting descriptors::

        class FluidLayerConfig(BaseSettings):
            fps = Setting(float, 110.0)
            fluid_flow = Child(FluidFlowConfig)
            visualisation = Child(VisualisationFieldConfig)

    Each instance gets its own fresh child (created during BaseSettings.__init__).
    Access: ``config.fluid_flow.speed``.
    Children are not replaceable — mutate their fields instead.
    """

    def __init__(self, settings_type: type[C], description: str = "") -> None:
        self._settings_type = settings_type
        self._description = description
        self._name: str | None = None

    def __set_name__(self, owner, name):
        self._name = name

    @overload
    def __get__(self, obj: None, objtype: type) -> Child[C]: ...
    @overload
    def __get__(self, obj: Any, objtype: type) -> C: ...
    def __get__(self, obj: Any | None, objtype: type | None = None) -> Child[C] | C:
        if obj is None:
            return self
        try:
            return obj._children[self._name]
        except (AttributeError, KeyError):
            raise AttributeError(
                f"'{type(obj).__name__}' child '{self._name}' not initialized"
            )

    def __set__(self, obj: Any, value: Any) -> None:
        raise AttributeError(
            f"Cannot replace child '{self._name}' — mutate its fields instead"
        )

    @property
    def name(self):
        return self._name

    @property
    def settings_type(self):
        return self._settings_type

    @property
    def description(self):
        return self._description
