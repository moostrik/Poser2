"""LutGrade - a texture drawn through a 3D LUT, the `.cube` files of a folder, picked on the panel."""

from __future__ import annotations
import logging
from pathlib import Path

from modules.gl import Texture, shaders
from modules.settings import Field, BaseSettings, Widget

logger = logging.getLogger(__name__)


class LutSettings(BaseSettings):
    """LUT color grading configuration."""
    folder:   Field[str]       = Field("data/luts", access=Field.INIT, description="LUT folder")
    files:    Field[list[str]] = Field([""], visible=False, access=Field.READWRITE)
    file:     Field[str]       = Field("", widget=Widget.text_select, options=files, description="LUT file", access=Field.READWRITE)
    rescan:   Field[bool]      = Field(False, widget=Widget.button, description="Rescan LUTs")
    strength: Field[float]     = Field(1.0, min=0.0, max=1.0)


class LutGrade:
    """Draws a texture through the LUT `settings.file` names, at its `strength`, or as it is without one; keeps the
    panel's list of the folder's LUTs."""

    def __init__(self, settings: LutSettings) -> None:
        self.settings: LutSettings = settings
        self._lut_shader: shaders.Lut = shaders.Lut()
        self._current_lut: str = ""
        self._scan_luts()
        self.settings.bind(LutSettings.folder, self._on_folder)
        self.settings.bind(LutSettings.rescan, self._on_rescan)

    def allocate(self) -> None:
        self._lut_shader.allocate()
        self._load_lut_if_changed()

    def deallocate(self) -> None:
        self._lut_shader.deallocate()
        self.settings.unbind(LutSettings.folder, self._on_folder)
        self.settings.unbind(LutSettings.rescan, self._on_rescan)

    @property
    def active(self) -> bool:
        """Whether a LUT is loaded and drawn."""
        return self._lut_shader.lut_loaded and bool(self.settings.file) and self.settings.strength > 0

    def draw(self, texture: Texture) -> None:
        """`texture` into the bound target, through the LUT where one is active."""
        self._load_lut_if_changed()
        if self.active:
            self._lut_shader.use(texture, self.settings.strength)
        else:
            texture.draw()

    def _on_folder(self, _: object) -> None:
        self._scan_luts()

    def _on_rescan(self, value: object) -> None:
        if value:
            self._scan_luts()

    def _scan_luts(self) -> None:
        """Scan lut folder for .cube files and update files list. Resets selection if no longer valid."""
        lut_dir = Path(self.settings.folder)
        if lut_dir.exists():
            filenames = [f.name for f in sorted(lut_dir.glob("*.cube"))]
        else:
            logger.warning("LUT directory not found: %s", lut_dir)
            filenames = []
        self.settings.files = [""] + filenames
        if self.settings.file and self.settings.file not in filenames:
            self.settings.file = ""

    def _load_lut_if_changed(self) -> None:
        """Load LUT file if selection changed."""
        if self.settings.file != self._current_lut:
            self._current_lut = self.settings.file
            if self.settings.file:
                lut_path = Path(self.settings.folder) / self.settings.file
                self._lut_shader.load_cube(str(lut_path))
