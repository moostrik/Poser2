"""Child descriptor for declaring nested Settings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from modules.settings.field import Field

if TYPE_CHECKING:
    from modules.settings.settings import Settings


class Child:
    """Descriptor for declaring child Settings on a parent Settings class.

    Single child (default, count=1) — accessed directly::

        player = Child(PlayerSettings)
        # cam.player.folder

    Multiple children — accessed by index::

        cores = Child(CoreSettings, count=3, share=[fps, color])
        # cam.cores[0].exposure

    ``count`` accepts an int literal or a Field reference::

        num   = Field(3, access=Field.INIT)
        cores = Child(CoreSettings, count=num, share=[fps])

    ``share`` takes a list of Field descriptors declared on the *parent*.
    Their values are copied to each child at construction time (passed
    as kwargs).  Shared fields are excluded from the child's
    serialization — the parent is the single source of truth.
    """

    def __init__(
        self,
        settings_type: type[Settings],
        *,
        count: int | Field = 1,
        share: list[Field] | None = None,
    ):
        self.settings_type = settings_type
        self.count = count          # int literal or Field descriptor
        self._share_refs: list[Field] = list(share) if share else []
        self.share: list[str] = []  # resolved lazily via _resolve_share()
        self.name = ""

    def __set_name__(self, owner, name):
        self.name = name
        # __set_name__ is called after the class body finishes, so Field
        # descriptors now have their .name set — safe to resolve.
        self.share = [f.name for f in self._share_refs]

    # -- Descriptor protocol -------------------------------------------------

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self                         # class-level access
        children = obj._children[self.name]
        if self.is_single:
            return children                     # single Settings instance
        return tuple(children)                  # immutable snapshot of the list

    def __set__(self, obj, value):
        raise AttributeError(
            f"Cannot replace child '{self.name}' — mutate its fields instead"
        )

    # -- Helpers used by Settings.__init__ -----------------------------------

    @property
    def is_single(self) -> bool:
        """True when count is the literal int 1 (direct access, not a list)."""
        return isinstance(self.count, int) and self.count == 1

    def resolve_count(self, owner: Settings) -> int:
        """Resolve count to an integer.

        If count is a Field descriptor, read its current value from *owner*.
        """
        if isinstance(self.count, int):
            return self.count
        # Field descriptor — read its value from the owner instance
        return self.count.__get__(owner, type(owner))

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
