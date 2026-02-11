"""Automatic GUI frame generation from ConfigBase classes.

Temporary bridge to existing PyReallySimpleGui until NiceGUI migration.
Generates frames with automatic two-way binding.
"""

from enum import Enum
from typing import Any, Callable
from modules.gui.PyReallySimpleGui import Gui, eType as eT, Element as E, Frame
from modules.ConfigBase import ConfigBase


class ConfigGuiGenerator:
    """Automatically generates GUI frames from ConfigBase with two-way binding.

    Example:
        >>> config = MyConfig()
        >>> gui_gen = ConfigGuiGenerator(config, gui, "Settings")
        >>> gui.addFrame(gui_gen.frame)
        >>> # Changes in config automatically update GUI
        >>> # Changes in GUI automatically update config
    """

    def __init__(self, config: ConfigBase, gui: Gui, name: str, elements_per_row: int = 3) -> None:
        """Initialize GUI generator for a config.

        Args:
            config: ConfigBase instance to generate GUI for
            gui: Gui instance
            name: Name for the GUI frame
            elements_per_row: Number of field controls per row (default: 2)
        """
        self.config = config
        self.gui = gui
        self.name = name
        self.elements_per_row = elements_per_row

        # Generate GUI elements and group them into rows
        elements = self._generate_elements()
        gui_height = gui.calculate_frame_height(elements)
        self.frame = Frame(name, elements, gui_height)

        # Setup two-way binding: config changes → GUI updates
        self._setup_watchers()

    def _generate_elements(self) -> list:
        """Generate GUI elements from config metadata.

        Groups multiple field controls into rows based on elements_per_row.
        """
        # First, collect all individual element pairs [label, input]
        all_element_pairs = []

        for field_name, metadata in self.config.info().items():
            # Skip fields with repr=False
            if not metadata.get("repr", True):
                continue
            element_pair = self._create_element_row(field_name, metadata)
            if element_pair:
                all_element_pairs.append(element_pair)

        # Group element pairs into rows
        grouped_rows = []
        current_row = []
        fields_in_row = 0

        for element_pair in all_element_pairs:
            # Add the elements to current row
            current_row.extend(element_pair)
            fields_in_row += 1

            # Check if we've reached elements_per_row field controls
            if fields_in_row >= self.elements_per_row:
                grouped_rows.append(current_row)
                current_row = []
                fields_in_row = 0

        # Add remaining elements as final row
        if current_row:
            grouped_rows.append(current_row)

        return grouped_rows

    def _create_element_row(
        self,
        field_name: str,
        metadata: dict[str, Any]
    ) -> list | None:
        """Create a GUI element row for a field."""
        key = f"{self.name}_{field_name}"

        # Skip fixed fields - set at init only, no runtime interaction
        if metadata["fixed"]:
            return None

        field_type = metadata["type"]
        label = metadata["label"]
        value = metadata["value"]
        min_val = metadata.get("min")
        max_val = metadata.get("max")

        # Enum → Combo box
        if isinstance(field_type, type) and issubclass(field_type, Enum):
            choices = [e.name for e in field_type]
            return [E(eT.TEXT, label), E(eT.CMBO, key, self._make_setter(field_name, field_type), value.name if isinstance(value, Enum) else value, choices)]

        # Boolean → Checkbox (label integrated in checkbox)
        if field_type == bool:
            return [E(eT.CHCK, key, self._make_setter(field_name, bool), value, text=label)]

        # Numeric with range → Slider
        elif field_type in (int, float) and min_val is not None and max_val is not None:
            resolution = 0.01 if field_type == float else 1
            return [E(eT.TEXT, label), E(eT.SLDR, key, self._make_setter(field_name, field_type), value, [min_val, max_val], resolution)]

        # Numeric without range → Input text
        elif field_type in (int, float):
            return [E(eT.TEXT, label), E(eT.ITXT, key, self._make_setter(field_name, field_type), str(value))]

        # String → Input text
        elif field_type == str:
            return [E(eT.TEXT, label), E(eT.ITXT, key, self._make_setter(field_name, str), value)]

        # Unsupported type
        else:
            return None

    def _make_setter(self, field_name: str, field_type: type) -> Callable[[Any], None]:
        """Create setter callback: GUI → Config with validation."""
        def setter(value: Any) -> None:
            try:
                # Type conversion
                if isinstance(field_type, type) and issubclass(field_type, Enum):
                    # Convert enum name back to enum member
                    converted = field_type[value]
                elif field_type == bool:
                    converted = bool(value)
                elif field_type == int:
                    converted = int(float(value))
                elif field_type == float:
                    converted = float(value)
                elif field_type == str:
                    converted = str(value)
                else:
                    converted = value

                setattr(self.config, field_name, converted)

            except (ValueError, TypeError, KeyError) as e:
                # Invalid input - revert GUI to current config value
                print(f"[ConfigGUI] Invalid input '{value}' for {field_name}: {e}")
                current_value = getattr(self.config, field_name)
                key = f"{self.name}_{field_name}"

                # Convert back to display format
                if isinstance(current_value, Enum):
                    display_value = current_value.name
                else:
                    display_value = str(current_value)

                self.gui.updateElement(key, display_value, useCallback=False)

        return setter

    def _setup_watchers(self) -> None:
        """Setup watchers: Config → GUI updates."""
        for field_name, metadata in self.config.info().items():
            # Skip fixed fields - they never change after init
            if metadata["fixed"]:
                continue
            # Skip fields with repr=False
            if not metadata.get("repr", True):
                continue

            key = f"{self.name}_{field_name}"

            def make_watcher(elem_key: str, field_type: type) -> Callable[[Any], None]:
                def watcher(value: Any) -> None:
                    # Convert enum to its name for GUI display
                    display_value = value.name if isinstance(value, Enum) else value
                    self.gui.updateElement(elem_key, display_value, useCallback=False)
                return watcher

            self.config.watch(make_watcher(key, metadata["type"]), field_name)
