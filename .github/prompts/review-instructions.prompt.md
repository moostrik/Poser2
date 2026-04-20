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
- Rules in the **wrong file** (exists but `applyTo` doesn't reach the files that were edited)

A proposed rule must:
- Address a mistake that would plausibly recur (not one-off or hypothetical)
- Apply across multiple modules or situations (not a single-site fix)
- Add information not already inferrable from existing rules
- Prefer what to do over what to avoid


Keep proposed rules and instructions brief — every line costs context budget on every future request.