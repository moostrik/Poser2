---
description: "Review whether project instructions covered the completed work."
---
Review the work just completed in this conversation against the relevant
instruction files (copilot-instructions.md and any .instructions.md files
that applied).

Flag only actionable findings:
- Rules that were **missing** and would have prevented a mistake or repeated correction
- Rules that were **wrong** or outdated
- Rules that were **redundant** across files (duplicated between always-on and targeted instructions)

Also flag quality issues in the instruction files themselves:
- Rules in the **wrong file** (exists but `applyTo` doesn't reach the files that were edited)
- **Negative rules** that restate a positive rule already present

Do not flag rules that worked correctly. Do not suggest additions for edge cases
that didn't actually cause problems. Proposed rules must be generalizable patterns
applicable across the project, not one-off facts about specific modules. Keep
instructions brief — every line costs context budget on every future request.
