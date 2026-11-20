# The rules' plan

What is still to do on the rules in `CLAUDE.md` and `.claude/rules/`. A built step leaves this plan.

## A. One rule per bullet

- A1. Go through the CLAUDE.md bullets that hold more than one rule, one at a time, with Matthias

## C. The Waker

The refactor plan builds `utils.Waker` (its B1), which replaces `utils.wait_until`: a thread waits on it until a moment, and a new item or stop wakes it with `set()`. Rule text only, one commit, when the Waker is committed and before the loops are moved to it.

- C1. `CLAUDE.md` Time: the line that goes makes place for the new one
  - goes: "Pace a loop with `utils.wait_until`; a wait's timeout (`Event.wait(t)`, `Queue.get(timeout=t)`) ends on Windows' 15.6 ms tick"
  - new: "A thread that waits for a moment waits on a `utils.Waker`; a wait's timeout (`Event.wait(t)`, `Queue.get(timeout=t)`) ends on Windows' 15.6 ms tick"

## Rules it touches

- `CLAUDE.md` "One rule per bullet": A
- `CLAUDE.md` Time: C
