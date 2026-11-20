# smile's plan

What is next for smile, the app in [SMILE.md](SMILE.md); what spans the apps is in the
[refactor's plan](../../REFACTOR_PLAN.md).

## Open questions

- Whether smile's 48 clip cells run from one thread with a heap of deadlines instead of a thread each
  (`ClipLoader`, `apps/smile/player/clip_loader.py`): 47 threads fewer and one Waker, against `CLAUDE.md`
  Concurrency, the part that works on the item has its own thread. Each cell waits on a Waker of its own first
  (REFACTOR_PLAN.md B1).

## The rules it touches

- `CLAUDE.md` Concurrency, the part that works on the item has its own thread: the open question
- `CLAUDE.md` Time, a thread that waits for a moment waits on a `utils.Waker`: the open question
- `CLAUDE.md` Plans and commits, a plan is named after the package, app or folder it is for, and sits in it: this
  plan, in `apps/smile`
