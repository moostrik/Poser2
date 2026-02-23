"""BaseSettings — collection of Setting descriptors with attribute access, callbacks, and serialization."""

import threading
import typing

from modules.settings.setting import Setting, Widget


class BaseSettings:
    """Collection of Setting descriptors with attribute access, callbacks, and serialization.

    Subclass and declare Setting descriptors as class attributes.
    Override init_only fields via constructor kwargs::

        class CameraSettings(BaseSettings):
            exposure = Setting(1000, min=100, max=10000)
            resolution = Setting(1080, init_only=True)

        settings = CameraSettings(resolution=720)

    Child settings are detected automatically from type annotations whose
    type is a BaseSettings subclass::

        class RenderSettings(BaseSettings):
            flow: FlowSettings          # auto-instantiated child
            fps = Setting(60.0)
    """

    def __init__(self, **kwargs):
        object.__setattr__(self, "_initialized", False)
        object.__setattr__(self, "_fields", {})
        object.__setattr__(self, "_values", {})
        object.__setattr__(self, "_callbacks", {})
        object.__setattr__(self, "_locks", {})
        object.__setattr__(self, "_children", {})

        # Collect Setting descriptors from the class hierarchy
        for cls in type(self).__mro__:
            for attr_name, attr_value in vars(cls).items():
                if isinstance(attr_value, Setting) and attr_name not in self._fields:
                    self._fields[attr_name] = attr_value
                    default = attr_value.default
                    self._values[attr_name] = list(default) if isinstance(default, list) else default
                    self._callbacks[attr_name] = []
                    self._locks[attr_name] = threading.Lock()

        # Auto-detect children from annotations: any annotated type that is a
        # BaseSettings subclass (and not already a Setting descriptor) becomes
        # a child, auto-instantiated with defaults.
        for cls in type(self).__mro__:
            annotations = vars(cls).get("__annotations__", {})
            for attr_name, ann in annotations.items():
                if attr_name in self._fields or attr_name in self._children:
                    continue
                # Resolve string annotations if needed
                resolved = ann
                if isinstance(ann, str):
                    ns = {**vars(typing), **{c.__name__: c for c in type(self).__mro__ if hasattr(c, '__name__')}}
                    # Also pull the module globals where the class was defined
                    import sys
                    mod = sys.modules.get(cls.__module__, None)
                    if mod:
                        ns.update(vars(mod))
                    try:
                        resolved = eval(ann, ns)
                    except Exception:
                        continue
                if isinstance(resolved, type) and issubclass(resolved, BaseSettings) and resolved is not BaseSettings:
                    self._children[attr_name] = resolved()

        # Apply kwargs (init_only fields are still writable here)
        for name, value in kwargs.items():
            if name not in self._fields:
                raise TypeError(
                    f"{type(self).__name__}() got unexpected keyword argument '{name}'"
                )
            self._fields[name].set(self, value)

        object.__setattr__(self, "_initialized", True)

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
        # Children: return child instance
        try:
            children = object.__getattribute__(self, "_children")
            if name in children:
                return children[name]
        except AttributeError:
            pass
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
        """Read-only view of all Setting descriptors, keyed by name."""
        return dict(self._fields)

    @property
    def actions(self):
        """Read-only view of all Widget.button Settings, keyed by name."""
        return {n: f for n, f in self._fields.items() if f.widget == Widget.button}

    @property
    def children(self):
        """Read-only view of all Child instances, keyed by name."""
        return dict(self._children)

    # -- Callback registration -----------------------------------------------

    def bind(self, field: 'Setting', callback) -> None:
        """Register a callback for a Setting descriptor.

        Usage::

            settings.bind(MySettings.exposure, on_exposure_changed)
            settings.bind(MySettings.reset, on_reset_fired)
        """
        if field.name not in self._fields:
            raise KeyError(
                f"'{type(self).__name__}' has no setting '{field.name}'"
            )
        field.bind(self, callback)

    def unbind(self, field: 'Setting', callback) -> None:
        """Remove a previously registered callback for a Setting."""
        field.unbind(self, callback)

    def bind_all(self, callback) -> None:
        """Register *callback* on every Setting field in this instance."""
        for field in self._fields.values():
            field.bind(self, callback)

    def unbind_all(self, callback) -> None:
        """Remove *callback* from every Setting field in this instance."""
        for field in self._fields.values():
            field.unbind(self, callback)

    # -- Serialization -------------------------------------------------------

    def to_dict(self):
        """Serialize mutable fields to a dict (excludes readonly and init_only).

        Children are serialized as nested dicts.
        """
        result = {}
        for name, field in self._fields.items():
            if not field.readonly:
                result[name] = field.to_json_value(self)
        for name, child in self._children.items():
            result[name] = child.to_dict()
        return result

    def update_from_dict(self, data):
        """Restore fields from a dict. Skips init_only fields after init.

        Nested dicts matching child names are forwarded to the child.
        """
        for name, raw in data.items():
            if name in self._children and isinstance(raw, dict):
                self._children[name].update_from_dict(raw)
            elif name in self._fields:
                field = self._fields[name]
                if field.init_only and self._initialized:
                    continue
                field.from_json_value(self, raw)

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
