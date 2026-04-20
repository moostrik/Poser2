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

Keep instructions brief — every line costs context budget on every future request.

Before proposing any new rule, apply this filter:
- Did the absence of this rule cause a mistake that would plausibly recur?
  If not, skip it — do not suggest rules for one-off errors or hypothetical edge cases.
- Does the rule apply across multiple modules or situations? If it only
  prevents one specific bug in one specific place, it is too narrow.
- Could the rule be inferred from an existing rule? If yes, the existing
  rule is sufficient — do not add a restatement.
