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

Name-mapped sharing allows a parent field to be shared under a different
name in the child::

    frequency = Field(30.0)
    interp = InterpolatorSettings(share=[frequency.as_('input_frequency')])

The parent's ``frequency`` value is shared to the child as ``input_frequency``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from modules.settings.field import Field, FieldAlias

if TYPE_CHECKING:
    from modules.settings.settings import Settings


class Child:
    """Descriptor that declares a child Settings on a parent.

    Single child (no *count*)::

        player = PlayerSettings(share=[fps])

    Multiple children::

        cores = CoreSettings(count=num, share=[fps])

    ``count`` accepts an int literal or a Field reference.
    ``share`` accepts a list of Field descriptors (or FieldAlias) from the parent.

    Name-mapped sharing::

        frequency = Field(30.0)
        interp = InterpolatorSettings(share=[frequency.as_('input_frequency')])

    Users never instantiate ``Child`` directly — the Settings metaclass
    creates one when it detects ``share`` or ``count`` in the constructor
    kwargs.
    """

    def __init__(
        self,
        settings_type: type,
        *,
        count: int | Field | None = None,
        share: list[Field | FieldAlias] | None = None,
    ):
        self.settings_type = settings_type
        self.is_list: bool = count is not None
        self.count = count
        self._share_refs: list[Field | FieldAlias] = list(share) if share else []
        self.share: list[str] = []  # parent field names
        self.share_map: dict[str, str] = {}  # parent_name → child_name
        self.name = ""

    def __set_name__(self, owner, name):
        self.name = name
        # Resolve share Field refs → names (Fields have .name set by now)
        # Build both the parent names list and the parent→child mapping
        for ref in self._share_refs:
            if isinstance(ref, FieldAlias):
                parent_name = ref.field.name
                child_name = ref.child_name
            else:
                parent_name = ref.name
                child_name = ref.name  # same name
            self.share.append(parent_name)
            self.share_map[parent_name] = child_name

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
        for parent_name in self.share:
            child_name = self.share_map[parent_name]
            if parent_name not in owner._fields:
                raise TypeError(
                    f"{type(owner).__name__}.{self.name}: shared field '{parent_name}' "
                    f"not found on parent {type(owner).__name__}"
                )
            if child_name not in child_fields:
                raise TypeError(
                    f"{type(owner).__name__}.{self.name}: shared field '{child_name}' "
                    f"not found on child {self.settings_type.__name__}"
                )
            parent_type = owner._fields[parent_name].type_
            child_type = child_fields[child_name].type_
            if parent_type != child_type:
                raise TypeError(
                    f"{type(owner).__name__}.{self.name}: type mismatch for shared field "
                    f"'{parent_name}' → '{child_name}' — parent {parent_type.__name__} != child {child_type.__name__}"
                )

    def build_share_kwargs(self, owner: Settings) -> dict:
        """Build constructor kwargs from the parent's shared field values.

        Keys are child field names (which may differ from parent names when
        using ``field.as_('child_name')``).
        """
        kwargs = {}
        for parent_name in self.share:
            child_name = self.share_map[parent_name]
            if parent_name in owner._fields:
                kwargs[child_name] = owner._values[parent_name]
        return kwargs
