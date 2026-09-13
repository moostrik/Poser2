---
description: Writing and updating design documents (CALIBRATION.md, TRACKING.md, LAYERS.md, STATES.md)
paths:
  - "apps/**/*.md"
  - "modules/**/*.md"
---
# Design documents

## Content

- Give each document one audience and purpose: `CALIBRATION.md` the operator's procedure and what explains a failed step, `TRACKING.md` the tracker's design, `STATES.md` the choreography, `LAYERS.md` the layers
- State a component's design reasoning in that component's document; other documents link to it
- State how the system works now, why it works that way, and the data behind it
- Put measured and derived numbers in tables, and name the configuration they assume (preset, resolution, tilt)
- Leave out how a conclusion was reached, what the code used to do, and what changed; that belongs in commit messages
- Keep a rejected alternative only when its reason still constrains the design, and state it as a constraint, not a story
- Mark anything not verified in code: **(site fact)** when told by the operator, **(deduction)** when inferred
- Collect open questions in one "Open" section per document

## Structure

- State each fact in one place; elsewhere, link to that section instead of restating it
- Don't restate what a module docstring owns; link to the module
- Put operator procedures before reference material
- Refer to code by file and symbol, not line number; line numbers go stale

## Tables

- Align the columns in the source so the table reads in a plain-text editor
- Keep cells short and consistent within a column; detail belongs in the text

## Updating

- Edit the section that states the fact; never append a paragraph about the change
- After an update the document reads as if the current design had always been the design
- When condensing, keep every reason, number and table; cut only repetition, narrative and filler
- When renaming or removing a section, state or setting, search the documents for references to it
- Write plain declarative sentences, without emphasis phrases
