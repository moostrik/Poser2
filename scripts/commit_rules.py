"""The files of a commit beside the rule files that govern them.

A rule file in `.claude/rules/` governs the files its `paths` match; one without `paths`, and `CLAUDE.md`,
govern every file. With `--hook` the list goes out as the JSON a Claude Code hook hands to Claude, with the
request to check the commit against those rules and report; the hook runs after every shell command and
answers only when the command it reads on stdin held a `git commit`. Below the rules stands pyright's result over
the repository, from its two runs: the oak4 venv with `pyproject.toml`, the main venv with `pyright.oak.json`.

    python scripts/commit_rules.py [commit] [--hook]
"""

import io
import json
import os
import re
import subprocess
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent.parent
RULES: Path = ROOT / ".claude" / "rules"
SHOWN: int = 6      # the files named per rule file; the rest are counted
COMMIT: re.Pattern[str] = re.compile(r"\bgit\s+(?:-\S+\s+(?:[^-\s]\S*\s+)?)*commit\b")   # git, its options, commit
REQUEST: str = ("Check this commit against the rules below and report to the user, brief: "
                "one line per rule that is not met, with its file; one line when all are met. "
                "Report the pyright result in one line.")
PYTHON: str = "Scripts/python.exe" if os.name == "nt" else "bin/python"
PYRIGHT_RUNS: list[tuple[str, list[str]]] = [(".venv-oak4", []), (".venv", ["-p", "pyright.oak.json"])]
PYRIGHT_TIMEOUT: float = 120.0  # seconds, per run
PYRIGHT_SHOWN: int = 30         # the errors listed; the rest are counted


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


def pyright() -> str:
    """Pyright's result over the repository: clean with its file count and time, its errors, or why it did not run."""
    start: float = time.perf_counter()
    processes: list[tuple[str, subprocess.Popen[str]]] = []
    try:
        for venv, args in PYRIGHT_RUNS:
            python: Path = ROOT / venv / PYTHON
            if not python.is_file():
                return f"pyright: not run (no {venv}/{PYTHON})"
            processes.append((venv, subprocess.Popen([str(python), "-m", "pyright", "--outputjson", *args], cwd=ROOT,
                                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                                     text=True, encoding="utf-8")))
        files: int = 0
        errors: list[str] = []
        for venv, process in processes:
            try:
                stdout, stderr = process.communicate(timeout=max(0.0, start + PYRIGHT_TIMEOUT - time.perf_counter()))
            except subprocess.TimeoutExpired:
                return f"pyright: not run ({venv} timed out after {PYRIGHT_TIMEOUT:.0f} s)"
            try:
                output = json.loads(stdout)
                files += int(output["summary"]["filesAnalyzed"])
                for diagnostic in output["generalDiagnostics"]:
                    if diagnostic["severity"] != "error":
                        continue
                    file: Path = Path(diagnostic["file"]).resolve()
                    path: str = file.relative_to(ROOT).as_posix() if file.is_relative_to(ROOT) else str(file)
                    line: int = int(diagnostic["range"]["start"]["line"]) + 1
                    message: str = " ".join(str(diagnostic["message"]).split())
                    rule: str = f" ({diagnostic['rule']})" if "rule" in diagnostic else ""
                    errors.append(f"{path}:{line}: {message}{rule}")
            except (ValueError, KeyError, TypeError):
                reason: str = (stderr.strip().splitlines() or ["unreadable output"])[-1]
                return f"pyright: not run ({venv}: {reason})"
    except OSError as error:
        return f"pyright: not run ({error})"
    finally:
        for _, process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()
    if not errors:
        return f"pyright: clean ({files} files, {time.perf_counter() - start:.0f} s)"
    more: list[str] = [f"... and {len(errors) - PYRIGHT_SHOWN} more"] if len(errors) > PYRIGHT_SHOWN else []
    return "\n".join([f"pyright: {len(errors)} errors", *errors[:PYRIGHT_SHOWN], *more])


def held_commit() -> bool:
    """Whether the command a hook reads on stdin held a `git commit`; True without a command to read."""
    if sys.stdin.isatty():
        return True
    try:
        command = json.loads(sys.stdin.read())["tool_input"]["command"]
    except (ValueError, KeyError, TypeError):
        return True
    return COMMIT.search(str(command)) is not None


def main() -> None:
    assert __doc__ is not None
    parser: ArgumentParser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("commit", nargs="?", default="HEAD")
    parser.add_argument("--hook", action="store_true", help="print the JSON a Claude Code PostToolUse hook returns")
    args = parser.parse_args()
    if args.hook and not held_commit():
        return
    text: str = f"{report(args.commit)}\n\n{pyright()}"
    if args.hook:
        text = json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": text}})
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
