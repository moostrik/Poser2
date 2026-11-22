"""The request for the link to a plan file after the plan changed.

With `--hook` it reads the JSON a Claude Code hook hands on stdin after an `Edit` or `Write`; when the edited file
is named `*_PLAN.md` it prints the JSON that asks Claude to give the user the link to the file in the thread before
the turn ends. Any other file, or stdin it cannot read, gives no output. It refuses nothing.

    python scripts/plan_page.py --hook
"""

import io
import json
import sys
from argparse import ArgumentParser
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent.parent
SUFFIX: str = "_PLAN.md"
REQUEST: str = "This plan changed. Give the user the link to this file in the thread before this turn ends."


def edited_file() -> Path | None:
    """The file the hook's JSON on stdin names; None without one to read."""
    if sys.stdin.isatty():
        return None
    try:
        file_path = json.loads(sys.stdin.read())["tool_input"]["file_path"]
    except (ValueError, KeyError, TypeError):
        return None
    return Path(str(file_path)) if file_path else None


def plan_name(file: Path) -> str:
    """The plan's path from the repo's root, or as given when it lies outside the repo."""
    resolved: Path = file.resolve()
    return resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else str(file)


def main() -> None:
    assert __doc__ is not None
    parser: ArgumentParser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hook", action="store_true", help="print the JSON a Claude Code PostToolUse hook returns")
    args = parser.parse_args()
    if not args.hook:
        parser.print_help()
        return
    file: Path | None = edited_file()
    if file is None or not file.name.endswith(SUFFIX):
        return
    text: str = json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse",
                                                   "additionalContext": f"{plan_name(file)}: {REQUEST}"}})
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
