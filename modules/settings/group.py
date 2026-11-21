"""Group, the descriptor that declares a child settings instance on a parent (SETTINGS.md, Groups and sharing)."""

from typing import TYPE_CHECKING, Any, Generic, Mapping, TypeVar, overload, Sequence

from .field import Field, FieldAlias, Access

if TYPE_CHECKING:
    from .base_settings import BaseSettings

T = TypeVar("T")


class Group(Generic[T]):
    """The descriptor that declares a child settings instance on a parent.

    Parameters
    ----------
    settings_type : type[T]
        The settings class the child is an instance of.
    share : list of Field or FieldAlias, optional
        Parent fields whose values are shared with the child.  Sharing is
        bidirectional: parent writes propagate to the child, and child
        writes propagate upward to the parent (and fan out to siblings).
    last : bool, optional
        The child comes after the other children, a subclass's included,
        on a panel and in a settings file.
    description : str, optional
        One short line of panel text: what the child is. A GUI hint, as a
        field's description.
    tint : str, optional
        The child's panel's background, a CSS color, in place of its depth's
        grey; empty for the grey. A GUI hint.
    after : str, optional
        For a root group: the card (its dotted path from the root) its children
        follow as cards, in that card's tab; empty for a tab per child. A GUI hint.
    defaults : mapping of field name to value, optional
        The child's own fields' starting values in this place, in place of
        their class defaults (a shared field takes the parent's value). The
        field's own ``default``, which the panel shows, stays the class's.
    """

    def __init__(
        self,
        settings_type: type[T],
        *,
        share: Sequence[Field | FieldAlias] | None = None,
        last: bool = False,
        description: str = "",
        tint: str = "",
        after: str = "",
        defaults: Mapping[str, Any] | None = None,
    ):
        self.settings_type = settings_type
        self.last: bool = last
        self.description: str = description
        self.tint: str = tint
        self.after: str = after
        self.defaults: dict[str, Any] = dict(defaults) if defaults else {}
        self._share_refs: list[Field | FieldAlias] = list(share) if share else []
        self.share_map: dict[str, str] = {}  # parent_name → child_name
        self.name = ""

    def __set_name__(self, owner: type, name: str):
        self.name = name
        for ref in self._share_refs:
            if isinstance(ref, FieldAlias):
                parent_name = ref.field.name
                child_name = ref.child_name
            else:
                parent_name = ref.name
                child_name = ref.name
            self.share_map[parent_name] = child_name

    @overload
    def __get__(self, obj: None, objtype: type) -> 'Group[T]': ...
    @overload
    def __get__(self, obj: 'BaseSettings', objtype: type) -> T: ...

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj._children[self.name]

    def __set__(self, obj: 'BaseSettings', value):
        raise AttributeError(
            f"Cannot replace group '{self.name}' — mutate its fields instead"
        )

    # -- Helpers used by Settings.__init__ -----------------------------------

    def validate_wiring(self, owner: 'BaseSettings') -> None:
        """Check that every shared field exists on both parent and child
        with matching types and compatible access levels.

        Raises ``TypeError`` if:

        - a parent field name is not found on the parent
        - a child field name is not found on the child
        - the field types don't match
        - one side is ``INIT`` and the other is not (INIT must be symmetric)
        """
        child_fields = {
            name: f for cls in self.settings_type.__mro__
            for name, f in vars(cls).items()
            if isinstance(f, Field)
        }

        for parent_name, child_name in self.share_map.items():
            if parent_name not in owner._fields:
                raise TypeError(
                    f"{type(owner).__name__}.{self.name}: share field '{parent_name}' "
                    f"not found on parent {type(owner).__name__}"
                )
            if child_name not in child_fields:
                raise TypeError(
                    f"{type(owner).__name__}.{self.name}: share field '{child_name}' "
                    f"not found on child {self.settings_type.__name__}"
                )
            parent_field = owner._fields[parent_name]
            child_field = child_fields[child_name]
            pt, ct = parent_field.type_, child_field.type_
            if not (issubclass(pt, ct) or issubclass(ct, pt)):
                raise TypeError(
                    f"{type(owner).__name__}.{self.name}: type mismatch for share field "
                    f"'{parent_name}' → '{child_name}' — parent {pt.__name__} "
                    f"!= child {ct.__name__}"
                )
            p_init = parent_field.access is Access.INIT
            c_init = child_field.access is Access.INIT
            if p_init != c_init:
                raise TypeError(
                    f"{type(owner).__name__}.{self.name}: access mismatch for share field "
                    f"'{parent_name}' → '{child_name}' — parent is "
                    f"{parent_field.access.name} but child is {child_field.access.name}. "
                    f"Shared fields must both be INIT or both be non-INIT."
                )

    def build_share_kwargs(self, owner: 'BaseSettings') -> dict:
        """Build constructor kwargs from the group's defaults and the parent's shared field values."""
        kwargs = dict(self.defaults)
        for parent_name, child_name in self.share_map.items():
            if parent_name in owner._fields:
                kwargs[child_name] = owner._values[parent_name]
        return kwargs
