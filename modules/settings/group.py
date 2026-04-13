"""Group descriptor for declaring nested Settings.

``Group`` is the single way to declare a child Settings on a parent::

    class PoseGroup(Settings):
        frequency = Field(30.0, access=Field.INIT)

        bbox  = Group(BboxFeature, push=[frequency])
        point = Group(PointFeature, push=[frequency])

    pose = PoseGroup()
    pose.bbox.smoother.frequency   # accesses the child's field

Push fields are passed as constructor kwargs to the child.  The parent
is the source of truth — pushed fields are excluded from child serialization,
re-propagated on ``update_from_dict``, and pushed downstream on normal parent
assignment.

Pull fields propagate in the opposite direction: the child is the source
of truth and changes flow upward to the parent.  Pull-only parent fields
are excluded from parent serialization and cannot be written directly.

A field appearing in both ``push`` and ``pull`` is bidirectional — either
side can write and changes propagate both ways.  The parent remains the
serialization source of truth for bidirectional fields.

Name-mapped sharing allows a parent field to arrive under a different name
in the child::

    frequency = Field(30.0)
    interp = Group(InterpolatorSettings, push=[frequency.as_('input_frequency')])
"""

from typing import TYPE_CHECKING, Generic, TypeVar, overload, Sequence

from modules.settings.field import Field, FieldAlias, Access

if TYPE_CHECKING:
    from modules.settings.base_settings import BaseSettings

T = TypeVar("T")


class Group(Generic[T]):
    """Descriptor that declares a child Settings on a parent.

    Parameters
    ----------
    settings_type : type[T]
        The Settings subclass to instantiate as the child.
    push : list of Field or FieldAlias, optional
        Parent fields whose values are pushed to the child at construction
        time, after ``update_from_dict``, and on normal runtime assignment.
    pull : list of Field or FieldAlias, optional
        Child fields whose values are pulled up to the parent.  Changes on
        the child propagate upward; pull-only parent fields cannot be
        written directly.
    """

    def __init__(
        self,
        settings_type: type[T],
        *,
        push: Sequence[Field | FieldAlias] | None = None,
        pull: Sequence[Field | FieldAlias] | None = None,
    ):
        self.settings_type = settings_type
        self._push_refs: list[Field | FieldAlias] = list(push) if push else []
        self._pull_refs: list[Field | FieldAlias] = list(pull) if pull else []
        self.push_map: dict[str, str] = {}  # parent_name → child_name
        self.pull_map: dict[str, str] = {}  # parent_name → child_name
        self.name = ""

    def __set_name__(self, owner: type, name: str):
        self.name = name
        for ref in self._push_refs:
            if isinstance(ref, FieldAlias):
                parent_name = ref.field.name
                child_name = ref.child_name
            else:
                parent_name = ref.name
                child_name = ref.name
            self.push_map[parent_name] = child_name
        for ref in self._pull_refs:
            if isinstance(ref, FieldAlias):
                parent_name = ref.field.name
                child_name = ref.child_name
            else:
                parent_name = ref.name
                child_name = ref.name
            self.pull_map[parent_name] = child_name

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
        """Check that every push/pull field exists on both parent and child with matching types and access modes.

        Raises ``TypeError`` if:

        - a parent field name is not found on the parent
        - a child field name is not found on the child
        - the field types don't match
        - push: the child field is ``INIT`` but the parent field is not
        - pull: the parent field is ``INIT`` but the child field is not
        """
        child_fields = {
            name: f for cls in self.settings_type.__mro__
            for name, f in vars(cls).items()
            if isinstance(f, Field)
        }

        def _check(label: str, mapping: dict[str, str], *, check_push: bool) -> None:
            for parent_name, child_name in mapping.items():
                if parent_name not in owner._fields:
                    raise TypeError(
                        f"{type(owner).__name__}.{self.name}: {label} field '{parent_name}' "
                        f"not found on parent {type(owner).__name__}"
                    )
                if child_name not in child_fields:
                    raise TypeError(
                        f"{type(owner).__name__}.{self.name}: {label} field '{child_name}' "
                        f"not found on child {self.settings_type.__name__}"
                    )
                parent_field = owner._fields[parent_name]
                child_field = child_fields[child_name]
                if parent_field.type_ != child_field.type_:
                    raise TypeError(
                        f"{type(owner).__name__}.{self.name}: type mismatch for {label} field "
                        f"'{parent_name}' → '{child_name}' — parent {parent_field.type_.__name__} "
                        f"!= child {child_field.type_.__name__}"
                    )
                if check_push:
                    if child_field.access is Access.INIT and parent_field.access is not Access.INIT:
                        raise TypeError(
                            f"{type(owner).__name__}.{self.name}: access mismatch for push field "
                            f"'{parent_name}' → '{child_name}' — child is INIT but parent is "
                            f"{parent_field.access.name}. An INIT child field cannot receive "
                            f"runtime updates from a non-INIT parent."
                        )
                else:
                    if parent_field.access is Access.INIT and child_field.access is not Access.INIT:
                        raise TypeError(
                            f"{type(owner).__name__}.{self.name}: access mismatch for pull field "
                            f"'{parent_name}' → '{child_name}' — parent is INIT but child is "
                            f"{child_field.access.name}. An INIT parent field cannot receive "
                            f"runtime updates from a non-INIT child."
                        )

        _check("push", self.push_map, check_push=True)
        _check("pull", self.pull_map, check_push=False)

    def build_push_kwargs(self, owner: 'BaseSettings') -> dict:
        """Build constructor kwargs from the parent's pushed field values."""
        kwargs = {}
        for parent_name, child_name in self.push_map.items():
            if parent_name in owner._fields:
                kwargs[child_name] = owner._values[parent_name]
        return kwargs
