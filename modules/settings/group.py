"""Group descriptor for declaring nested Settings.

``Group`` is the single way to declare a child Settings on a parent::

    class PoseGroup(Settings):
        frequency = Field(30.0, access=Field.INIT)

        bbox  = Group(BboxFeature, share=[frequency])
        point = Group(PointFeature, share=[frequency])

    pose = PoseGroup()
    pose.bbox.smoother.frequency   # accesses the child's field

Shared fields are passed as constructor kwargs to the child.  The parent
is the source of truth — shared fields are excluded from child serialization
and re-propagated on ``update_from_dict``.

Name-mapped sharing allows a parent field to arrive under a different name
in the child::

    frequency = Field(30.0)
    interp = Group(InterpolatorSettings, share=[frequency.as_('input_frequency')])
"""

from typing import TYPE_CHECKING, Generic, TypeVar, overload, Sequence

from modules.settings.field import Field, FieldAlias

if TYPE_CHECKING:
    from modules.settings.settings import Settings

T = TypeVar("T")


class Group(Generic[T]):
    """Descriptor that declares a child Settings on a parent.

    Parameters
    ----------
    settings_type : type[T]
        The Settings subclass to instantiate as the child.
    share : list of Field or FieldAlias, optional
        Parent fields whose values are pushed to the child at construction
        and after ``update_from_dict``.
    """

    def __init__(
        self,
        settings_type: type[T],
        *,
        share: Sequence[Field | FieldAlias] | None = None,
    ):
        self.settings_type = settings_type
        self._share_refs: list[Field | FieldAlias] = list(share) if share else []
        self.share_map: dict[str, str] = {} # parent_name → child_name
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
    def __get__(self, obj: 'Settings', objtype: type) -> T: ...

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj._children[self.name]

    def __set__(self, obj: 'Settings', value):
        raise AttributeError(
            f"Cannot replace group '{self.name}' — mutate its fields instead"
        )

    # -- Helpers used by Settings.__init__ -----------------------------------

    def validate_share(self, owner: 'Settings') -> None:
        """Check that every shared field exists on both parent and child with matching types."""
        child_fields = {
            name: f for cls in self.settings_type.__mro__
            for name, f in vars(cls).items()
            if isinstance(f, Field)
        }
        for parent_name, child_name in self.share_map.items():
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

    def build_share_kwargs(self, owner: 'Settings') -> dict:
        """Build constructor kwargs from the parent's shared field values."""
        kwargs = {}
        for parent_name, child_name in self.share_map.items():
            if parent_name in owner._fields:
                kwargs[child_name] = owner._values[parent_name]
        return kwargs
