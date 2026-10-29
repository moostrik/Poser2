"""Guard test for the glossary (`.claude/rules/glossary.md`): no identifier in `modules/` or a
finished app, and no key or stored name in that app's presets, uses a synonym a finished rename
replaced.

Reads the source with `tokenize` and checks NAME tokens only, so comments and strings may still
quote an old name. Tests are checked too; vendored code is skipped. Preset keys are checked at
every depth, and so are string values that are identifiers (the enum names a preset stores, such
as a layer playlist's entries); files starting with '.' are not presets. Each entry is one
finished concept; add an entry when a rename lands, and add an app to `ROOTS` and `PRESET_DIRS`
when it follows.
"""

import json
import re
import tokenize
import unittest
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APPS = ['white_space', 'hd_trio', 'deep_flow']
ROOTS = [ROOT / 'modules'] + [ROOT / 'apps' / app for app in APPS]
PRESET_DIRS = [ROOT / 'apps' / app / 'data' / 'settings' for app in APPS]
SKIPPED_DIRS = {'vimbapython', '__pycache__'}
# Stale preset keys the settings tree no longer has: the loader ignores them, and they stay as saved.
STALE_PRESET_KEYS = {
    ('apps/deep_flow/data/settings/3D_rev2.json', 'render.fluid3d.visualisation'),
}

BANNED: list[tuple[str, str]] = [
    ('settings, not config', r'^_?config$'),
    ('track_id, not tracklet_id / pose_id', r'tracklet_ids?$|^pose_ids?$'),
    # not 'time_stamped': the verb, as in a motion time stamped from the first frame
    ('timestamp, not time_stamp', r'time_stamps?(?![a-z])'),
    ('num_cams, not num_cameras', r'num_cameras'),
    ('max_players, not max_poses / num_tracks', r'max_poses|num_tracks'),
    ('AngleVelocity, not AngleVel', r'AngleVel(?!ocity)'),
    ('AngleSymmetry, not AngleSym', r'AngleSym(?!metry)'),
    ('Confidence, not Conf', r'Conf(?=[A-Z_]|$)'),
    ('Ema, not EMA', r'EMA'),
    ('center, not centre', r'[Cc]entre'),
    ('visualization, not visualisation', r'[Vv]isualis'),
    ('Foreground, not Frg', r'Frg|frg'),
    ('DenseFlow, not DFlow', r'DFlow|Dlow'),
    ('MotionTime, not MTime', r'MTime'),
    ('Temperature, not TEMP', r'TEMP_BRIDGE'),
    ('HasSequencerState, not HasSequence', r'^HasSequence$|SequenceStoreMixin'),
    # not a bare 'ip_address': that is the standard library's ipaddress.ip_address, which net_probe calls
    ('ip, not ip_address', r'ip_address(?:es|_)|_ip_address'),
    ('hwaccel, not hwdecode', r'hwdecode'),
    ('add/remove_<kind>_callback, not register/unregister_callback', r'^register_callback$|^unregister_callback$'),
]


def source_files() -> list[Path]:
    return [path for root in ROOTS for path in sorted(root.rglob('*.py'))
            if not SKIPPED_DIRS.intersection(path.relative_to(ROOT).parts)]


def names(path: Path) -> list[tuple[int, str]]:
    """Every NAME token of a file with its line."""
    with tokenize.open(path) as f:
        return [(tok.start[0], tok.string) for tok in tokenize.generate_tokens(f.readline) if tok.type == tokenize.NAME]


def preset_files() -> list[Path]:
    return [path for folder in PRESET_DIRS for path in sorted(folder.glob('*.json')) if not path.name.startswith('.')]


def stored_names(node: object, path: str = '') -> Iterator[tuple[str, str]]:
    """Every key of a JSON tree and every identifier-like string value, at every depth, with
    the dotted path of the key it sits at."""
    if isinstance(node, dict):
        for key, value in node.items():
            key_path = f"{path}.{key}" if path else key
            yield key_path, key
            yield from stored_names(value, key_path)
    elif isinstance(node, list):
        for value in node:
            yield from stored_names(value, path)
    elif isinstance(node, str) and node.isidentifier():
        yield path, node


class TestGlossary(unittest.TestCase):

    def test_no_identifier_uses_a_replaced_synonym(self) -> None:
        patterns = [(concept, re.compile(regex)) for concept, regex in BANNED]
        found: list[str] = []
        for path in source_files():
            for lineno, name in names(path):
                for concept, pattern in patterns:
                    if pattern.search(name):
                        found.append(f"{path.relative_to(ROOT).as_posix()}:{lineno} {name}  ({concept})")
        self.assertEqual(found, [], "identifiers that use a replaced synonym:\n" + '\n'.join(found))

    def test_no_preset_name_uses_a_replaced_synonym(self) -> None:
        patterns = [(concept, re.compile(regex)) for concept, regex in BANNED]
        found: list[str] = []
        for path in preset_files():
            preset = path.relative_to(ROOT).as_posix()
            for key_path, name in stored_names(json.loads(path.read_text(encoding='utf-8'))):
                if (preset, key_path) in STALE_PRESET_KEYS:
                    continue
                for concept, pattern in patterns:
                    if pattern.search(name):
                        found.append(f"{preset} {key_path}: {name}  ({concept})")
        self.assertEqual(found, [], "preset keys or stored names that use a replaced synonym:\n" + '\n'.join(found))

    def test_every_preset_folder_has_presets(self) -> None:
        for folder in PRESET_DIRS:
            self.assertTrue(any(folder.glob('*.json')), f"no presets in {folder.relative_to(ROOT).as_posix()}")


if __name__ == '__main__':
    unittest.main()
