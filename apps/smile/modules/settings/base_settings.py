"""
Settings — a reactive dataclass.

Declare fields as class attributes; use them as plain variables.
Each field is a single line that can optionally carry GUI hints,
access restrictions, and value constraints — but these are declarations,
not code. The functional class that uses Settings never deals with GUI,
serialization, or threading logic.

    class CameraSettings(Settings):
        exposure = Field(1000, min=100, max=10000)
        resolution = Field(1080, access=Field.INIT)
        manual = Field(False)

    settings = CameraSettings(resolution=720)
    settings.exposure = 500          # set like a normal variable
    x = settings.exposure            # get like a normal variable
    settings.bind(CameraSettings.exposure, on_changed)  # react to changes

Group children nest related fields into scopes::

    class PoseGroup(Settings):
        frequency = Field(30.0, access=Field.INIT)
        bbox      = Group(BboxFeature, share=[frequency])
        point     = Group(PointFeature, share=[frequency])
"""

import copy
import logging
import threading

from .field import Field, Access
from .group import Group
from .widget import Widget

logger = logging.getLogger(__name__)


class BaseSettings:
    """Reactive dataclass — declare fields and groups as class attributes."""

    def __init__(self, **kwargs):
        object.__setattr__(self, "_initialized", False)
        object.__setattr__(self, "_parent", None)
        object.__setattr__(self, "_share_down", {})     # parent_name → [(child, child_name), ...]
        object.__setattr__(self, "_share_up", {})        # child_name → (parent, parent_name)
        object.__setattr__(self, "_is_propagating", False)
        object.__setattr__(self, "_fields", {})
        object.__setattr__(self, "_values", {})
        object.__setattr__(self, "_callbacks", {})
        object.__setattr__(self, "_locks", {})
        object.__setattr__(self, "_children", {})

        # Collect Field and Group descriptors from the class hierarchy
        group_names: dict[str, None] = {}
        for cls in type(self).__mro__:
            for attr_name, attr_value in vars(cls).items():
                if attr_name.startswith("_"):
                    continue
                if isinstance(attr_value, Field) and attr_name not in self._fields:
                    self._fields[attr_name] = attr_value
                    default = attr_value.default
                    self._values[attr_name] = copy.deepcopy(default) if isinstance(default, list) else default
                    self._callbacks[attr_name] = []
                    self._locks[attr_name] = threading.Lock()
                elif isinstance(attr_value, Group) and attr_name not in group_names:
                    group_names[attr_name] = None

        # Apply kwargs (init_only fields are still writable here)
        for name, value in kwargs.items():
            if name not in self._fields:
                raise TypeError(
                    f"{type(self).__name__}() got unexpected keyword argument '{name}'"
                )
            self._fields[name].set(self, value)

        # Create children from Group descriptors (after kwargs so shared fields have updated values)
        for attr_name in group_names:
            group_desc = self._get_group_descriptor(attr_name)
            if group_desc is None:
                continue
            group_desc.validate_wiring(self)
            share_kwargs = group_desc.build_share_kwargs(self)
            child = group_desc.settings_type(**share_kwargs)
            object.__setattr__(child, "_parent", self)
            # Wire sharing: both directions
            for parent_name, child_name in group_desc.share_map.items():
                self._share_down.setdefault(parent_name, []).append((child, child_name))
                child._share_up[child_name] = (self, parent_name)
            self._children[attr_name] = child

    # -- Attribute access guard ----------------------------------------------

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        if name in self._children:
            raise AttributeError(
                f"Cannot replace child '{name}' — mutate its fields instead"
            )
        # Allow @property setters defined on the class
        for cls in type(self).__mro__:
            if name in cls.__dict__ and isinstance(cls.__dict__[name], property):
                cls.__dict__[name].fset(self, value)
                return
        if name not in self._fields:
            raise AttributeError(
                f"'{type(self).__name__}' has no setting '{name}'"
            )
        self._fields[name].set(self, value)

    def __getattr__(self, name):
        # Children are accessed via Group descriptors — they don't reach here.
        # Fallback — normally the descriptor __get__ handles known fields.
        fields = object.__getattribute__(self, "_fields")
        if name in fields:
            return object.__getattribute__(self, "_values")[name]
        raise AttributeError(
            f"'{type(self).__name__}' has no setting '{name}'"
        )

    def __delattr__(self, name):
        if name.startswith("_"):
            object.__delattr__(self, name)
            return
        raise AttributeError(
            f"Cannot delete setting '{name}'"
        )

    # -- Field access --------------------------------------------------------

    @property
    def fields(self):
        """Read-only view of all Field descriptors, keyed by name."""
        return dict(self._fields)

    @property
    def actions(self):
        """Read-only view of all Widget.button fields, keyed by name."""
        return {n: f for n, f in self._fields.items() if f.widget == Widget.button}

    @property
    def parent(self) -> 'BaseSettings | None':
        """The parent Settings that owns this child, or None for root."""
        return self._parent

    @property
    def children(self):
        """Read-only view of all child instances, keyed by name."""
        return dict(self._children)

    def is_shared(self, name: str) -> bool:
        """Return True if *name* is shared from the parent into this instance."""
        return name in self._share_up

    def shared_source(self, name: str) -> str | None:
        """Return the parent field name that shares *name*, or None if local."""
        upward = self._share_up.get(name)
        return upward[1] if upward is not None else None

    def is_share_source(self, name: str) -> bool:
        """Return True if *name* shares its value to one or more children."""
        return name in self._share_down

    def wiring_label(self, name: str) -> str | None:
        """Return a short human-readable label describing the wiring of *name*.

        Returns ``None`` for locally-owned fields with no wiring.
        """
        if name in self._share_up:
            parent, parent_name = self._share_up[name]
            parent_type = type(parent).__name__
            return f"Shared from {parent_type}.{parent_name}"
        if name in self._share_down:
            targets = self._share_down[name]
            child_to_attr = {id(c): a for a, c in self._children.items()}
            parts = []
            for child, child_name in targets:
                attr = child_to_attr.get(id(child), "?")
                parts.append(f"{attr}.{child_name}")
            return f"Shares to {', '.join(parts)}"
        return None

    # -- Callback registration -----------------------------------------------

    def bind(self, field: 'Field', callback) -> None:
        """Register a callback for a field.

        Usage::

            settings.bind(MySettings.exposure, on_exposure_changed)
            settings.bind(MySettings.reset, on_reset_fired)
        """
        if field.name not in self._fields:
            raise KeyError(
                f"'{type(self).__name__}' has no setting '{field.name}'"
            )
        field.bind(self, callback)

    def unbind(self, field: 'Field', callback) -> None:
        """Remove a previously registered callback for a field."""
        field.unbind(self, callback)

    def bind_all(self, callback) -> None:
        """Register *callback* on every field in this instance."""
        for field in self._fields.values():
            field.bind(self, callback)

    def unbind_all(self, callback) -> None:
        """Remove *callback* from every field in this instance."""
        for field in self._fields.values():
            field.unbind(self, callback)

    # -- Serialization -------------------------------------------------------

    def to_dict(self, *, _exclude: set[str] | None = None):
        """Serialize mutable fields to a dict.

        Excludes ``Access.READ`` fields and ``Widget.button`` fields.
        ``Access.INIT`` fields **are** included (they are editable in JSON but
        cannot be changed at runtime after construction).

        Shared fields are excluded from child dicts (parent is source of
        truth for serialization).
        """
        result = {}
        exclude = _exclude or set()
        for name, field in self._fields.items():
            if name in exclude:
                continue
            if field.access is Access.READ or field.widget == Widget.button:
                continue
            result[name] = field.to_json_value(self)
        for name, child in self._children.items():
            desc = self._get_group_descriptor(name)
            # Exclude shared child fields (parent owns them)
            shared = set(desc.share_map.values()) if desc is not None else set()
            result[name] = child.to_dict(_exclude=shared)
        return result

    def initialize(self):
        """Lock INIT fields.  Call once after the startup preset is loaded.

        After this call, ``Access.INIT`` fields can no longer be written
        programmatically or via ``update_from_dict``.
        """
        object.__setattr__(self, '_initialized', True)
        for child in self._children.values():
            child.initialize()

    def update_from_dict(self, data):
        """Restore fields from a dict.

        ``Access.INIT`` fields are included only while ``_initialized``
        is ``False`` (before ``initialize()`` has been called).

        Nested dicts matching child names are forwarded to the child.
        Bad values are logged and skipped — one broken field does not
        prevent the remaining fields from loading.
        """
        # Pass 1: update own fields (so count fields like num_cameras are current)
        child_data = {}
        for name, raw in data.items():
            if name in self._children:
                child_data[name] = raw
            elif name in self._fields:
                field = self._fields[name]
                if field.access is Access.INIT and self._initialized:
                    continue
                try:
                    field.from_json_value(self, raw)
                except (TypeError, ValueError) as exc:
                    logger.warning(
                        "%s.update_from_dict: skipping '%s' — %s",
                        type(self).__name__, name, exc,
                    )
            else:
                logger.warning(
                    "%s.update_from_dict: ignoring unknown key '%s'",
                    type(self).__name__, name,
                )

        # Pass 2: forward child data
        for name, raw in child_data.items():
            child = self._children[name]
            if isinstance(raw, dict):
                child.update_from_dict(raw)
            else:
                logger.warning(
                    "%s.update_from_dict: expected dict for child '%s' — skipping",
                    type(self).__name__, name,
                )
        # Re-propagate shared fields to children after own fields are updated
        self._propagate_shared()

    # -- Share propagation ---------------------------------------------------

    def _propagate_shared(self):
        """Copy shared field values from this parent into its children.

        INIT fields are skipped when the child is already initialized — they
        were frozen by ``initialize()`` and cannot be changed at runtime.
        """
        for parent_name in self._share_down:
            self._propagate_shared_field(parent_name)

    def _propagate_shared_field(self, parent_name: str) -> None:
        """Push one parent field to all directly shared child fields.

        This is called after construction, after ``update_from_dict()``, and
        after normal runtime assignment so shared values stay reactive.
        Propagation is strictly downstream: parent → child.
        """
        if parent_name not in self._share_down or parent_name not in self._fields:
            return
        value = self._values[parent_name]
        for child, child_name in self._share_down[parent_name]:
            if child_name not in child._fields:
                continue
            child_field = child._fields[child_name]
            if child_field.access is Access.INIT and child._initialized:
                continue
            previous = child._is_propagating
            object.__setattr__(child, '_is_propagating', True)
            try:
                child_field.set(child, value)
            finally:
                object.__setattr__(child, '_is_propagating', previous)

    def _propagate_upward_field(self, child_name: str) -> None:
        """Propagate one child field value up to the parent.

        Called from ``Field._apply()`` on the child when a shared field changes.
        After writing the parent, the parent fans out the value to any other
        children that are wired to the same parent field.
        """
        upward = self._share_up.get(child_name)
        if upward is None:
            return
        parent, parent_name = upward
        if parent_name not in parent._fields:
            return
        parent_field = parent._fields[parent_name]
        if parent_field.access is Access.INIT and parent._initialized:
            return
        value = self._values[child_name]
        previous = parent._is_propagating
        object.__setattr__(parent, '_is_propagating', True)
        try:
            parent_field._apply(parent, value)
        finally:
            object.__setattr__(parent, '_is_propagating', previous)
        parent._propagate_upward_field(parent_name)

    # -- Helpers -------------------------------------------------------------

    def _get_group_descriptor(self, name: str) -> Group | None:
        """Look up a Group descriptor from the class hierarchy, or None."""
        for cls in type(self).__mro__:
            attr = vars(cls).get(name)
            if isinstance(attr, Group):
                return attr
        return None

    # -- Equality ------------------------------------------------------------

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        if self._values != other._values:
            return False
        for name, child in self._children.items():
            if child != other._children.get(name):
                return False
        return True

    __hash__: None = None  # mutable — unhashable

    # -- Repr ----------------------------------------------------------------

    def __repr__(self):
        parts = []
        for name, field in self._fields.items():
            if field.visible:
                value = self._values[name]
                parts.append(f"{name}={value!r}")
        for name, child in self._children.items():
            parts.append(f"{name}={type(child).__name__}(...)")
        class_name = type(self).__name__
        return f"{class_name}({', '.join(parts)})"
