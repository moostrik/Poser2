# The rules' plan

What is still to do on the rules in `CLAUDE.md` and `.claude/rules/`. A built step leaves this plan.

## A. One rule per bullet

- A1. Go through the CLAUDE.md bullets that hold more than one rule, one at a time, with Matthias

## B. The inference rules file

The package scan of `modules/inference` found design that no rule holds, and one spelling. Rule text only, one commit.

- B1. A new `.claude/rules/inference.md`, loaded for `modules/inference/**`, holds the lines under "New rules"
- B2. `CLAUDE.md` Settings: "metres" becomes "meters" (the glossary's American spelling)

### Rules that go

None.

### New rules

- "A reader inside the package lets its own stream wait for the item's event; the thread itself does not wait"
- "A tensor read on another stream than the one that made it is marked there (`record_stream`)"
- "A producer with readers inside and outside the package calls the inside ones first, then waits for its stream and calls its finished callbacks"
- "A model file's name says `_fp16` when it is FP16"

## Rules it touches

- `CLAUDE.md` "One rule per bullet": A
- `CLAUDE.md` "Put a rule where it loads when the code it governs is written": B
