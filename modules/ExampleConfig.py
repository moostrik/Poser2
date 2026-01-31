"""Example configuration demonstrating all ConfigBase features.

Showcases: normal/fixed fields, enums, ranges, types, metadata.
Demonstrates both field() and config_field() usage.
"""

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from modules.ConfigBase import ConfigBase, config_field


class RenderMode(Enum):
    """Rendering mode options."""
    WIREFRAME = "wireframe"
    SOLID = "solid"
    TEXTURED = "textured"


class Quality(IntEnum):
    """Quality level (IntEnum example)."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    ULTRA = 4


@dataclass
class ExampleConfig(ConfigBase):
    """Example config demonstrating all ConfigBase capabilities.

    Categories tested:
    - Normal fields: Can be modified at any time
    - Fixed fields: Set during __init__, then locked
    - Various types: bool, int, float, str, Enum, IntEnum
    - Range constraints: min/max for GUI sliders
    - Custom labels and descriptions
    """

    # NORMAL FIELDS - Can be changed at any time
    # Using config_field() - cleaner syntax
    enabled: bool = config_field(
        True,
        description="Enable/disable the feature",
        label="Feature Enabled"
    )

    strength: float = config_field(
        1.0,
        min=0.0,
        max=10.0,
        description="Processing strength (0=off, 10=max)",
        label="Strength"
    )

    threshold: int = config_field(
        50,
        min=0,
        max=100,
        description="Detection threshold percentage"
    )

    # Using field() - standard dataclass syntax
    name: str = field(
        default="Default",
        metadata={"description": "Configuration name"}
    )

    render_mode: RenderMode = field(
        default=RenderMode.SOLID,
        metadata={"description": "Visual rendering style"}
    )

    quality: Quality = field(
        default=Quality.MEDIUM,
        metadata={"description": "Processing quality level"}
    )

    # FIXED FIELDS - Can be set at init, then become locked
    # Using config_field() for cleaner fixed field syntax
    device_id: int = config_field(
        0,
        fixed=True,
        description="Hardware device ID (set at startup)"
    )

    buffer_size: int = config_field(
        1024,
        fixed=True,
        min=256,
        max=4096,
        description="Buffer size in bytes (cannot change after init)"
    )

    # NUMERIC WITHOUT RANGE - Shows as input text field
    custom_value: float = field(
        default=42.5,
        metadata={"description": "Custom numeric value (no range constraint)"}
    )

    def setup_watchers(self) -> None:
        """Setup watchers to print value changes.

        Demonstrates listener optimization:
        - Global watcher: Fires on ANY field change
        - Specific watchers: Fire ONLY when their field changes

        Example: When 'enabled' changes:
        - Global watcher fires → prints "Something changed!"
        - Enabled watcher fires → prints "Enabled: True/False"
        - Strength/render_mode watchers DO NOT fire (optimization!)
        """
        # Watch all changes
        self.watch(lambda: print(f"[ExampleConfig] Something changed!"))

        # Watch specific fields (only fire when their field changes)
        self.watch(lambda val: print(f"[ExampleConfig] Enabled: {val}"), 'enabled')
        self.watch(lambda val: print(f"[ExampleConfig] Strength: {val}"), 'strength')
        self.watch(lambda val: print(f"[ExampleConfig] Threshold: {val}"), 'threshold')
        self.watch(lambda val: print(f"[ExampleConfig] Name: {val}"), 'name')
        self.watch(lambda val: print(f"[ExampleConfig] Render mode: {val.name}"), 'render_mode')
        self.watch(lambda val: print(f"[ExampleConfig] Quality: {val.name}"), 'quality')
        self.watch(lambda val: print(f"[ExampleConfig] Custom value: {val}"), 'custom_value')
