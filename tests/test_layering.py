"""Guard test for the import rules of `modules/`: no app code, no camera backend outside the
camera package, and no reaching past another package's `__init__.py`.

Reads the source with `ast`, so annotation-only imports under TYPE_CHECKING count too.
Package tests may import their own package's internal files; vendored code is skipped.
"""

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = ROOT / 'modules'
CAMERA_BACKENDS = {'oak', 'oak4', 'replay', 'vimba', 'webcam'}
SKIPPED_DIRS = {'tests', 'vimbapython', '__pycache__'}


def source_files() -> list[Path]:
    return [path for path in sorted(MODULES.rglob('*.py'))
            if not SKIPPED_DIRS.intersection(path.relative_to(ROOT).parts)]


def imported_modules(path: Path) -> list[tuple[int, str]]:
    """Every module a file imports, relative imports resolved to their full dotted name."""
    package = list(path.relative_to(ROOT).with_suffix('').parts[:-1])
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'), filename=str(path))):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[:len(package) - node.level + 1]
                name = '.'.join(base + ([node.module] if node.module else []))
            else:
                name = node.module or ''
            found.append((node.lineno, name))
    return found


def violations(rule) -> list[str]:
    lines: list[str] = []
    for path in source_files():
        own_package = path.relative_to(MODULES).parts[0] if len(path.relative_to(MODULES).parts) > 1 else ''
        for lineno, name in imported_modules(path):
            parts = name.split('.')
            if rule(own_package, path, parts):
                lines.append(f"{path.relative_to(ROOT).as_posix()}:{lineno} imports {name}")
    return lines


class TestLayering(unittest.TestCase):

    def test_modules_never_import_apps(self) -> None:
        found = violations(lambda own, path, parts: parts[0] == 'apps')
        self.assertEqual(found, [], "modules/ imports app code:\n" + '\n'.join(found))

    def test_camera_backends_are_imported_only_inside_the_camera_package(self) -> None:
        found = violations(lambda own, path, parts: own != 'camera' and parts[:2] == ['modules', 'camera']
                           and len(parts) > 2 and parts[2] in CAMERA_BACKENDS)
        self.assertEqual(found, [], "a camera backend is imported outside modules/camera:\n" + '\n'.join(found))

    def test_other_packages_are_imported_through_their_init(self) -> None:
        found = violations(lambda own, path, parts: parts[0] == 'modules' and len(parts) > 2 and parts[1] != own)
        self.assertEqual(found, [], "an import reaches past another package's __init__.py:\n" + '\n'.join(found))


if __name__ == '__main__':
    unittest.main()
