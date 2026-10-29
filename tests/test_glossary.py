"""Guard test for the glossary (`.claude/rules/glossary.md`): no identifier in `modules/` uses a
synonym a finished rename replaced.

Reads the source with `tokenize` and checks NAME tokens only, so comments and strings may still
quote an old name. Tests are checked too; vendored code is skipped. Each entry is one finished
concept; add an entry when a rename lands, and widen `ROOTS` to `apps/` when the apps follow.
"""

import re
import tokenize
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOTS = [ROOT / 'modules']
SKIPPED_DIRS = {'vimbapython', '__pycache__'}

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


if __name__ == '__main__':
    unittest.main()
