---
description: Writing and updating documents
paths:
  - "apps/**/*.md"
  - "modules/**/*.md"
---
# Documents

These rules hold for documents, not for plans. A plan follows `.claude/rules/plans.md`.

## Content

- A document states the what, the how and the why
- A document is short and concise
- Avoid repetition and essays
- Give each document one audience and purpose
- A component is described in its own document; other documents link to it
- Leave out how a conclusion was reached, what the code used to do, and what changed; that belongs in commit messages
- Keep a rejected alternative only when its reason still constrains the design, and state it as a constraint, not a story
- Mark anything not verified in code: **(site fact)** when someone told it, **(deduction)** when inferred
- Put open questions and unbuilt work in the package's or app's plan, not in the document
- Write plain declarative sentences, without emphasis phrases

## Structure

- State each fact in one place; elsewhere, link to that section instead of restating it
- Don't restate what a module docstring owns; link to the module
- Put procedures before reference material
- Refer to code by file and symbol, not line number; line numbers go stale

## Tables

- A table gives an overview, and may repeat the text for it
- Put measured and derived numbers in tables, and name the configuration they assume (preset, resolution)
- A measurement is costly to get: keep it, and ask before deleting an outdated one
- Align the columns in the source so the table reads in a plain-text editor
- Keep cells short and consistent within a column; detail belongs in the text

## Updating

- Edit the section that states the fact; never append a paragraph about the change
- When condensing, keep every reason and table; cut only repetition, narrative and filler
- When renaming or removing a section, state or setting, search the documents for references to it
