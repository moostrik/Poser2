# The rules' plan

What is still to do on the rules in `CLAUDE.md` and `.claude/rules/`. A built step leaves this plan.

## A. One rule per bullet

- A1. Go through the CLAUDE.md bullets that hold more than one rule, one at a time, with Matthias

## B. Pyright in the commit check

Type errors built up unnoticed (565 before the sweep of 1e5e9c8a). The repository is clean outside the tests now, and a full run takes about 16 s, so every commit can show whether it stays so, also when a change in one file breaks another.

- B1. `scripts/commit_rules.py` runs pyright over the repository after a commit and adds the result to what it hands over: "pyright: clean", or the errors. Two runs, side by side: `.venv-oak4/Scripts/python.exe -m pyright` (the config in `pyproject.toml`) and `.venv/Scripts/python.exe -m pyright -p pyright.oak.json` (`modules/camera/oak`). It reports and refuses nothing; where a venv or pyright is missing it says so in one line. Checked by running the script on a clean tree and on a tree with one type error put in for the check
- B2. `CLAUDE.md` Types: "A commit leaves no pyright errors"

## Rules it touches

- `CLAUDE.md` "One rule per bullet": A
- `CLAUDE.md` Types: B
