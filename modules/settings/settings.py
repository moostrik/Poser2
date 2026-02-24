"""Settings — collection of Field descriptors with attribute access, callbacks, and serialization."""

import copy
import logging
import sys
import threading
import typing

from modules.settings.field import Field, Access
from modules.settings.widget import Widget

logger = logging.getLogger(__name__)


class Settings:
    """Collection of Field descriptors with attribute access, callbacks, and serialization.

    Subclass and declare Field descriptors as class attributes.
    Override init_only fields via constructor kwargs::

        class CameraSettings(Settings):
            exposure = Field(1000, min=100, max=10000)
            resolution = Field(1080, access=Field.INIT)

        settings = CameraSettings(resolution=720)

    Child settings are declared via type annotations::

        class RenderSettings(Settings):
            flow: FlowSettings          # auto-instantiated child
            fps = Field(60.0)
    """

    def __init__(self, **kwargs):
        object.__setattr__(self, "_initialized", False)
        object.__setattr__(self, "_fields", {})
        object.__setattr__(self, "_values", {})
        object.__setattr__(self, "_callbacks", {})
        object.__setattr__(self, "_locks", {})
        object.__setattr__(self, "_children", {})

        # Collect Field descriptors and child Settings from the class hierarchy
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
                elif (
                    isinstance(attr_value, type)
                    and issubclass(attr_value, Settings)
                    and attr_value is not Settings
                    and attr_name not in self._fields
                    and attr_name not in self._children
                ):
                    child = attr_value()
                    self._children[attr_name] = child
                    object.__setattr__(self, attr_name, child)

        # Also detect children from type annotations
        for cls in type(self).__mro__:
            annotations = vars(cls).get("__annotations__", {})
            for attr_name, ann in annotations.items():
                if attr_name in self._fields or attr_name in self._children:
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
                    child = resolved()
                    self._children[attr_name] = child
                    object.__setattr__(self, attr_name, child)

        # Apply kwargs (init_only fields are still writable here)
        for name, value in kwargs.items():
            if name not in self._fields:
                raise TypeError(
                    f"{type(self).__name__}() got unexpected keyword argument '{name}'"
                )
            self._fields[name].set(self, value)

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
    def children(self):
        """Read-only view of all child instances, keyed by name."""
        return dict(self._children)

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

    def to_dict(self):
        """Serialize mutable fields to a dict.

        Excludes ``Access.READ`` fields and ``Widget.button`` fields.
        ``Access.INIT`` fields **are** included (they are editable in JSON but
        cannot be changed at runtime after construction).

        Children are serialized as nested dicts.
        """
        result = {}
        for name, field in self._fields.items():
            if field.access is Access.READ or field.widget == Widget.button:
                continue
            result[name] = field.to_json_value(self)
        for name, child in self._children.items():
            result[name] = child.to_dict()
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
        for name, raw in data.items():
            if name in self._children and isinstance(raw, dict):
                self._children[name].update_from_dict(raw)
            elif name in self._children:
                logger.warning(
                    "%s.update_from_dict: expected dict for child '%s', got %s — skipping",
                    type(self).__name__, name, type(raw).__name__,
                )
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
