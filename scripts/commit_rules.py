"""The files of a commit beside the rule files that govern them.

A rule file in `.claude/rules/` governs the files its `paths` match; one without `paths`, and `CLAUDE.md`,
govern every file. With `--hook` the list goes out as the JSON a Claude Code hook hands to Claude, with the
request to check the commit against those rules and report.

    python scripts/commit_rules.py [commit] [--hook]
"""

import io
import json
import re
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent.parent
RULES: Path = ROOT / ".claude" / "rules"
SHOWN: int = 6      # the files named per rule file; the rest are counted
REQUEST: str = ("Check this commit against the rules below and report to the user: "
                "per rule file, what is met and what is not, with the file it is in.")


def git(*args: str) -> str:
    """The output of a git command run at the repo's root."""
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True).stdout


def commit_files(commit: str) -> list[str]:
    """The files a commit changed, as paths from the repo's root."""
    names: str = git("show", "--name-only", "--format=", "--no-renames", commit)
    return [line for line in names.splitlines() if line]


def rule_paths(rule_file: Path) -> list[str] | None:
    """The `paths` of a rule file's frontmatter; None when it has none and so governs every file."""
    lines: list[str] = rule_file.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    paths: list[str] | None = None
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if line.startswith("paths:"):
            paths = []
        elif paths is not None and line.lstrip().startswith("- "):
            paths.append(line.lstrip()[2:].strip().strip("\"'"))
        elif paths is not None and line and not line.startswith(" "):
            break
    return paths


def pattern_regex(pattern: str) -> re.Pattern[str]:
    """A `paths` glob as a regex: `**` crosses folders, `*` and `?` stay inside one."""
    regex: str = ""
    i: int = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            regex += "(?:.*/)?"
            i += 3
        elif pattern.startswith("**", i):
            regex += ".*"
            i += 2
        elif pattern[i] == "*":
            regex += "[^/]*"
            i += 1
        elif pattern[i] == "?":
            regex += "[^/]"
            i += 1
        else:
            regex += re.escape(pattern[i])
            i += 1
    return re.compile(regex)


def governed(files: list[str]) -> dict[str, list[str]]:
    """Each rule file that governs a file of the commit, with the files it governs."""
    result: dict[str, list[str]] = {"CLAUDE.md": files}
    for rule_file in sorted(RULES.glob("**/*.md")):
        paths: list[str] | None = rule_paths(rule_file)
        if paths is None:
            matched: list[str] = files
        else:
            regexes: list[re.Pattern[str]] = [pattern_regex(path) for path in paths]
            matched = [file for file in files if any(regex.fullmatch(file) for regex in regexes)]
        if matched:
            result[rule_file.relative_to(ROOT).as_posix()] = matched
    return result


def report(commit: str) -> str:
    """The commit's line, the request, and each rule file with the files it governs."""
    files: list[str] = commit_files(commit)
    lines: list[str] = [f"Commit {git('log', '-1', '--format=%h %s', commit).strip()}", REQUEST, ""]
    for rule_file, matched in governed(files).items():
        if matched is files:
            lines.append(f"{rule_file}: every file ({len(files)})")
        else:
            more: str = f" and {len(matched) - SHOWN} more" if len(matched) > SHOWN else ""
            lines.append(f"{rule_file}: {', '.join(matched[:SHOWN])}{more}")
    return "\n".join(lines)


def main() -> None:
    parser: ArgumentParser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("commit", nargs="?", default="HEAD")
    parser.add_argument("--hook", action="store_true", help="print the JSON a Claude Code PostToolUse hook returns")
    args = parser.parse_args()
    text: str = report(args.commit)
    if args.hook:
        text = json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": text}})
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
