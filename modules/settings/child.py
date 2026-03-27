"""Child descriptor for declaring nested Settings.

Created automatically when a Settings subclass is called with ``share=``
or ``count=`` kwargs — users never write ``Child(...)`` directly::

    class CameraSettings(Settings):
        num    = Field(3, access=Field.INIT)
        fps    = Field(30.0, access=Field.INIT)
        player = PlayerSettings(share=[fps])
        cores  = CoreSettings(count=num, share=[fps])

    cam = CameraSettings()
    cam.player.folder           # single instance
    cam.cores[0].exposure       # list, indexed access
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from modules.settings.field import Field

if TYPE_CHECKING:
    from modules.settings.settings import Settings


class Child:
    """Descriptor that declares a child Settings on a parent.

    Single child (no *count*)::

        player = PlayerSettings(share=[fps])

    Multiple children::

        cores = CoreSettings(count=num, share=[fps])

    ``count`` accepts an int literal or a Field reference.
    ``share`` accepts a list of Field descriptors from the parent.

    Users never instantiate ``Child`` directly — the Settings metaclass
    creates one when it detects ``share`` or ``count`` in the constructor
    kwargs.
    """

    def __init__(
        self,
        settings_type: type,
        *,
        count: int | Field | None = None,
        share: list[Field] | None = None,
    ):
        self.settings_type = settings_type
        self.is_list: bool = count is not None
        self.count = count
        self._share_refs: list[Field] = list(share) if share else []
        self.share: list[str] = []
        self.name = ""

    def __set_name__(self, owner, name):
        self.name = name
        # Resolve share Field refs → names (Fields have .name set by now)
        self.share = [f.name for f in self._share_refs]

    # -- Descriptor protocol -------------------------------------------------

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        children = obj._children[self.name]
        if self.is_list:
            return tuple(children)
        return children

    def __set__(self, obj, value):
        raise AttributeError(
            f"Cannot replace child '{self.name}' — mutate its fields instead"
        )

    # -- Helpers used by Settings.__init__ -----------------------------------

    def resolve_count(self, owner: Settings) -> int:
        """Resolve count to an integer.

        If count is a Field descriptor, read its current value from *owner*.
        """
        if isinstance(self.count, int):
            return self.count
        if isinstance(self.count, Field):
            return self.count.__get__(owner, type(owner))
        raise TypeError(f"{self.name}: count is None — cannot resolve")

    def validate_share(self, owner: Settings) -> None:
        """Check that every shared field exists on both parent and child with matching types."""
        child_fields = {
            name: f for cls in self.settings_type.__mro__
            for name, f in vars(cls).items()
            if isinstance(f, Field)
        }
        for field_name in self.share:
            if field_name not in owner._fields:
                raise TypeError(
                    f"{type(owner).__name__}.{self.name}: shared field '{field_name}' "
                    f"not found on parent {type(owner).__name__}"
                )
            if field_name not in child_fields:
                raise TypeError(
                    f"{type(owner).__name__}.{self.name}: shared field '{field_name}' "
                    f"not found on child {self.settings_type.__name__}"
                )
            parent_type = owner._fields[field_name].type_
            child_type = child_fields[field_name].type_
            if parent_type != child_type:
                raise TypeError(
                    f"{type(owner).__name__}.{self.name}: type mismatch for shared field "
                    f"'{field_name}' — parent {parent_type.__name__} != child {child_type.__name__}"
                )

    def build_share_kwargs(self, owner: Settings) -> dict:
        """Build constructor kwargs from the parent's shared field values."""
        kwargs = {}
        for field_name in self.share:
            if field_name in owner._fields:
                kwargs[field_name] = owner._values[field_name]
        return kwargs
