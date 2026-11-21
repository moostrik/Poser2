"""BaseSettings, the reactive data container every settings class derives from (SETTINGS.md)."""

import copy
import logging
import threading
from typing import Any

from .field import Field, Access
from .group import Group
from .widget import Widget

logger = logging.getLogger(__name__)


class BaseSettings:
    """A reactive data container: its `Field` and `Group` class attributes are its values and its children."""

    def __init__(self, **kwargs):
        object.__setattr__(self, "_initialized", False)
        object.__setattr__(self, "_share_down", {})     # parent_name → [(child, child_name), ...]
        object.__setattr__(self, "_share_up", {})        # child_name → (parent, parent_name)
        object.__setattr__(self, "_fields", {})
        object.__setattr__(self, "_values", {})
        object.__setattr__(self, "_callbacks", {})
        object.__setattr__(self, "_locks", {})
        object.__setattr__(self, "_children", {})

        # Collect Field and Group descriptors from the class hierarchy, base classes first, so a
        # panel and a settings file list a base's fields before a subclass's; a name a subclass declares
        # again keeps the base's place and takes the subclass's descriptor
        mro: tuple[type, ...] = type(self).__mro__
        names: dict[str, None] = {}
        for cls in reversed(mro):
            for attr_name, attr_value in vars(cls).items():
                if not attr_name.startswith("_") and isinstance(attr_value, (Field, Group)):
                    names.setdefault(attr_name, None)
        group_names: dict[str, Group] = {}
        for attr_name in names:
            attr_value = next(vars(cls)[attr_name] for cls in mro if attr_name in vars(cls))
            if isinstance(attr_value, Field):
                self._fields[attr_name] = attr_value
                default = attr_value.default
                self._values[attr_name] = copy.deepcopy(default) if isinstance(default, list) else default
                self._callbacks[attr_name] = []
                self._locks[attr_name] = threading.Lock()
            elif isinstance(attr_value, Group):
                group_names[attr_name] = attr_value

        # Apply kwargs (init_only fields are still writable here)
        for name, value in kwargs.items():
            if name not in self._fields:
                raise TypeError(
                    f"{type(self).__name__}() got unexpected keyword argument '{name}'"
                )
            self._fields[name].set(self, value)

        # Create children from Group descriptors (after kwargs so shared fields have updated values); a group
        # declared `last` after the others, a subclass's included
        for attr_name in sorted(group_names, key=lambda name: group_names[name].last):     # stable: declared order kept
            group_desc = self._get_group_descriptor(attr_name)
            if group_desc is None:
                continue
            group_desc.validate_wiring(self)
            share_kwargs = group_desc.build_share_kwargs(self)
            child = group_desc.settings_type(**share_kwargs)
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
        field = self._fields.get(name)
        if field is not None:
            field.set(self, value)
            return
        # A name the class declares with a setter (a property, a Group, which refuses a replacement): Python's own
        if hasattr(getattr(type(self), name, None), "__set__"):
            object.__setattr__(self, name, value)
            return
        raise AttributeError(
            f"'{type(self).__name__}' has no setting '{name}'"
        )

    def __getattr__(self, name: str) -> Any:
        """A name that is no field, group or attribute: refused."""
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
    def fields(self) -> dict[str, Field]:
        """Read-only view of all Field descriptors, keyed by name."""
        return dict(self._fields)

    @property
    def actions(self) -> dict[str, Field]:
        """Read-only view of all Widget.button fields, keyed by name."""
        return {n: f for n, f in self._fields.items() if f.widget == Widget.button}

    @property
    def children(self) -> dict[str, 'BaseSettings']:
        """Read-only view of all child instances, keyed by name."""
        return dict(self._children)

    def is_shared(self, name: str) -> bool:
        """Return True if *name* is shared from the parent into this instance."""
        return name in self._share_up

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
        """Register *callback* for *field* (`MySettings.exposure`), called with each new value."""
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

    def to_dict(self, *, _exclude: set[str] | None = None) -> dict[str, Any]:
        """The tree as plain values: every field but `READ` ones, buttons and `saved=False` ones, a child's shared
        fields left to its parent."""
        result = {}
        exclude = _exclude or set()
        for name, field in self._fields.items():
            if name in exclude:
                continue
            if field.access is Access.READ or field.widget == Widget.button or not field.saved:
                continue
            result[name] = field.to_json_value(self)
        for name, child in self._children.items():
            desc = self._get_group_descriptor(name)
            # Exclude shared child fields (parent owns them)
            shared = set(desc.share_map.values()) if desc is not None else set()
            result[name] = child.to_dict(_exclude=shared)
        return result

    def initialize(self) -> None:
        """Lock the tree's `INIT` fields: from here a write raises and a load skips them."""
        object.__setattr__(self, '_initialized', True)
        for child in self._children.values():
            child.initialize()

    def update_from_dict(self, data) -> None:
        """Set the fields *data* holds, own fields first, then each child's; an unknown key or a bad value is
        logged and skipped."""
        # Pass 1: update own fields (so count fields like num_cams are current)
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

    def _propagate_shared(self) -> None:
        """Every shared field's value pushed down into the children."""
        for parent_name in self._share_down:
            self._propagate_shared_field(parent_name)

    def _propagate_shared_field(self, parent_name: str) -> None:
        """One shared field's value pushed down into the children it is shared to; a child's `INIT` field after
        `initialize()` keeps its own."""
        if parent_name not in self._share_down or parent_name not in self._fields:
            return
        value = self._values[parent_name]
        for child, child_name in self._share_down[parent_name]:
            child_field = child._fields.get(child_name)
            if child_field is None or (child_field.access is Access.INIT and child._initialized):
                continue
            child_field._apply(child, value, from_parent=True)

    def _propagate_upward_field(self, child_name: str) -> None:
        """A shared field's value written into the parent it is shared from, which shares it on with its other
        children and its own parent; a parent's `INIT` field after `initialize()` keeps its own."""
        upward = self._share_up.get(child_name)
        if upward is None:
            return
        parent, parent_name = upward
        parent_field = parent._fields.get(parent_name)
        if parent_field is None or (parent_field.access is Access.INIT and parent._initialized):
            return
        parent_field._apply(parent, self._values[child_name])

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
