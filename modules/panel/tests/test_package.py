"""The app's side of the panel loads no NiceGUI: NiceGUI runs in the panel's own process."""

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


class TestImport(unittest.TestCase):
    def test_import_loads_no_nicegui(self):
        code = "import modules.panel, sys; assert 'nicegui' not in sys.modules, 'nicegui loaded'"
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
