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

Child settings group related fields into nested scopes::

    class CameraSettings(Settings):
        num    = Field(3, access=Field.INIT)
        fps    = Field(30.0, access=Field.INIT)
        player = PlayerSettings(share=[fps])              # single child
        cores  = CoreSettings(count=num, share=[fps])     # N children
"""

import copy
import logging
import sys
import threading
import typing

from modules.settings.child import Child
from modules.settings.field import Field, Access
from modules.settings.widget import Widget

logger = logging.getLogger(__name__)


class SettingsMeta(type):
    """Metaclass that intercepts ``share`` / ``count`` in constructor kwargs.

    When a Settings subclass is called with ``share=`` or ``count=`` it
    returns a :class:`Child` descriptor instead of a normal instance::

        player = PlayerSettings(share=[fps])          # → Child descriptor
        cores  = CoreSettings(count=num, share=[fps]) # → Child descriptor
        s      = PlayerSettings()                     # → normal instance
    """

    def __call__(cls, *args, **kwargs):
        if 'share' in kwargs or 'count' in kwargs:
            return Child(
                cls,
                count=kwargs.pop('count', None),
                share=kwargs.pop('share', None),
            )
        return super().__call__(*args, **kwargs)


class Settings(metaclass=SettingsMeta):
    """Reactive dataclass — declare fields and children as class attributes."""

    def __init__(self, **kwargs):
        object.__setattr__(self, "_initialized", False)
        object.__setattr__(self, "_parent", None)
        object.__setattr__(self, "_fields", {})
        object.__setattr__(self, "_values", {})
        object.__setattr__(self, "_callbacks", {})
        object.__setattr__(self, "_locks", {})
        object.__setattr__(self, "_children", {})

        # Collect Field and Child descriptors from the class hierarchy
        child_names: set[str] = set()
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
                elif isinstance(attr_value, Child) and attr_name not in child_names:
                    child_names.add(attr_name)
                elif (
                    isinstance(attr_value, type)
                    and issubclass(attr_value, Settings)
                    and attr_value is not Settings
                    and attr_name not in self._fields
                    and attr_name not in self._children
                    and attr_name not in child_names
                ):
                    # Legacy: bare Settings subclass as class attribute
                    child = attr_value()
                    object.__setattr__(child, "_parent", self)
                    self._children[attr_name] = child
                    object.__setattr__(self, attr_name, child)

        # Also detect children from type annotations (deprecated)
        for cls in type(self).__mro__:
            annotations = vars(cls).get("__annotations__", {})
            for attr_name, ann in annotations.items():
                if attr_name in self._fields or attr_name in self._children or attr_name in child_names:
                    continue
                resolved = ann
                if isinstance(ann, str):
                    ns = {**vars(typing)}
                    mod = sys.modules.get(cls.__module__, None)
                    if mod:
                        ns.update(vars(mod))
                    try:
                        resolved = eval(ann, ns)
                    except Exception:
                        continue
                if isinstance(resolved, type) and issubclass(resolved, Settings) and resolved is not Settings:
                    # logger.warning(
                    #     "%s: annotation-based child '%s' is deprecated — use Child() descriptor",
                    #     type(self).__name__, attr_name,
                    # )
                    child = resolved()
                    object.__setattr__(child, "_parent", self)
                    self._children[attr_name] = child
                    object.__setattr__(self, attr_name, child)

        # Apply kwargs (init_only fields are still writable here)
        for name, value in kwargs.items():
            if name not in self._fields:
                raise TypeError(
                    f"{type(self).__name__}() got unexpected keyword argument '{name}'"
                )
            self._fields[name].set(self, value)

        # Create children from Child descriptors (after kwargs so count fields have updated values)
        for attr_name in child_names:
            child_desc = self._get_child_descriptor(attr_name)
            if child_desc is None:
                continue
            child_desc.validate_share(self)
            share_kwargs = child_desc.build_share_kwargs(self)
            if child_desc.is_list:
                count = child_desc.resolve_count(self)
                children = []
                for _ in range(count):
                    child = child_desc.settings_type(**share_kwargs)
                    object.__setattr__(child, "_parent", self)
                    children.append(child)
                self._children[attr_name] = children
            else:
                child = child_desc.settings_type(**share_kwargs)
                object.__setattr__(child, "_parent", self)
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
        # Children are stored as instance attributes — they don't reach here.
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
    def parent(self) -> 'Settings | None':
        """The parent Settings that owns this child, or None for root."""
        return self._parent

    @property
    def children(self):
        """Read-only view of all child instances, keyed by name.

        List children are flattened: ``cores`` with 3 items becomes
        ``{'cores 0': c0, 'cores 1': c1, 'cores 2': c2}``.
        """
        out = {}
        for name, child_or_list in self._children.items():
            if isinstance(child_or_list, list):
                for i, child in enumerate(child_or_list):
                    out[f"{name} {i}"] = child
            else:
                out[name] = child_or_list
        return out

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

        Children are serialized as nested dicts.  Shared fields (declared
        via ``Child(..., share=[...])``) are excluded from child dicts —
        the parent is the single source of truth.  When name-mapped sharing
        is used (``field.as_('child_name')``), the child name is excluded.
        """
        result = {}
        exclude = _exclude or set()
        for name, field in self._fields.items():
            if name in exclude:
                continue
            if field.access is Access.READ or field.widget == Widget.button:
                continue
            result[name] = field.to_json_value(self)
        for name, child_or_list in self._children.items():
            try:
                child_desc = self._get_child_descriptor(name)
                # Use child names (share_map values) for exclusion
                shared = set(child_desc.share_map.values())
            except LookupError:
                shared = set()
            if isinstance(child_or_list, list):
                result[name] = [c.to_dict(_exclude=shared) for c in child_or_list]
            else:
                result[name] = child_or_list.to_dict(_exclude=shared)
        return result

    def initialize(self):
        """Lock INIT fields.  Call once after the startup preset is loaded.

        After this call, ``Access.INIT`` fields can no longer be written
        programmatically or via ``update_from_dict``.
        """
        object.__setattr__(self, '_initialized', True)
        for child_or_list in self._children.values():
            if isinstance(child_or_list, list):
                for child in child_or_list:
                    child.initialize()
            else:
                child_or_list.initialize()

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

        # Resize multi-children lists if their count field changed
        if not self._initialized:
            self._resize_children()

        # Pass 2: forward child data
        for name, raw in child_data.items():
            child_or_list = self._children[name]
            if isinstance(child_or_list, list) and isinstance(raw, list):
                for child, child_dict in zip(child_or_list, raw):
                    if isinstance(child_dict, dict):
                        child.update_from_dict(child_dict)
            elif not isinstance(child_or_list, list) and isinstance(raw, dict):
                child_or_list.update_from_dict(raw)
            else:
                logger.warning(
                    "%s.update_from_dict: type mismatch for child '%s' — skipping",
                    type(self).__name__, name,
                )
        # Re-propagate shared fields to children after own fields are updated
        self._propagate_shared()

    # -- Equality ------------------------------------------------------------

    def _resize_children(self):
        """Resize list children to match their descriptor's current count."""
        for name, child_or_list in self._children.items():
            if not isinstance(child_or_list, list):
                continue
            try:
                child_desc = self._get_child_descriptor(name)
            except LookupError:
                continue
            target = child_desc.resolve_count(self)
            if len(child_or_list) == target:
                continue
            share_kwargs = child_desc.build_share_kwargs(self)
            if len(child_or_list) < target:
                for _ in range(target - len(child_or_list)):
                    child = child_desc.settings_type(**share_kwargs)
                    object.__setattr__(child, "_parent", self)
                    child_or_list.append(child)
            else:
                del child_or_list[target:]

    def _propagate_shared(self):
        """Copy shared field values from this parent into its children."""
        if self._initialized:
            return
        for name, child_or_list in self._children.items():
            try:
                child_desc = self._get_child_descriptor(name)
            except LookupError:
                continue
            if not child_desc.share:
                continue
            share_kwargs = child_desc.build_share_kwargs(self)
            targets = child_or_list if isinstance(child_or_list, list) else [child_or_list]
            for child in targets:
                for field_name, value in share_kwargs.items():
                    if field_name in child._fields:
                        child._fields[field_name].set(child, value)

    # -- Helpers -------------------------------------------------------------

    def _get_child_descriptor(self, name: str) -> Child:
        """Look up a Child descriptor from the class hierarchy."""
        for cls in type(self).__mro__:
            attr = vars(cls).get(name)
            if isinstance(attr, Child):
                return attr
        raise LookupError(f"No Child descriptor '{name}' on {type(self).__name__}")

    # -- Equality ------------------------------------------------------------

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        if self._values != other._values:
            return False
        for name, child_or_list in self._children.items():
            other_child = other._children.get(name)
            if isinstance(child_or_list, list):
                if not isinstance(other_child, list) or len(child_or_list) != len(other_child):
                    return False
                for a, b in zip(child_or_list, other_child):
                    if a != b:
                        return False
            elif child_or_list != other_child:
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
        for name, child_or_list in self._children.items():
            if isinstance(child_or_list, list):
                type_name = type(child_or_list[0]).__name__ if child_or_list else '?'
                parts.append(f"{name}=[{type_name}(...) \u00d7 {len(child_or_list)}]")
            else:
                parts.append(f"{name}={type(child_or_list).__name__}(...)")
        class_name = type(self).__name__
        return f"{class_name}({', '.join(parts)})"
