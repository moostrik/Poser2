# The rules' plan

What is still to do on the rules in `CLAUDE.md` and `.claude/rules/`. A built step leaves this plan.

## A. One rule per bullet

- A1. Go through the CLAUDE.md bullets that hold more than one rule, one at a time, with Matthias

## B. A plan's page follows the plan

A plan is edited in its file on the PC; its page on claude.ai stayed old until Matthias asked. The notes that said to update the page were not read at the moment of the edit. A hook speaks at every edit, as the commit check does at every commit.

- B1. The hook. `.claude/settings.json` runs `scripts/plan_page.py --hook` after `Edit|Write`; for a file named `*_PLAN.md` it tells Claude: "This plan changed. Update its page on claude.ai before this turn ends, and give the user the link." It refuses nothing. Check: building B takes B out of this file, so the hook fires and this plan's page is updated without being asked
- B2. Notes that go, since the hook does their work:
  - project memory `republish-plan-on-change.md`
  - project memory `MEMORY.md`: "Every thread publishes its plan as an artifact page and republishes after each change"
  - project memory `threads-and-plans.md`: "Update a plan's artifact when the plan changes"
  - the PC's own memory: the lines on republishing after each edit in `watch-repo-file-not-attachments.md`, `flow-docs-layout.md`, `recording-plan-b.md`, `frame-path-plan.md`, `fluid-3d-refactor-plan.md`
- B3. Tell the threads through the coordinator: the session that edits a plan updates its page

## Rules it touches

- `CLAUDE.md` "One rule per bullet": A
- No rule: B is a hook in place of a rule
