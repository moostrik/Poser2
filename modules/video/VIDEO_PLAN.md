# The video module's plan

What is to be built in `modules/video`: ffmpeg, the clips and streams it writes and reads, and the screen recorder
([VIDEO.md](VIDEO.md)).

## C. Names

C2. A time stamp is `timestamp` in `screen_recorder.py`: `stamps`, `_stamps`, `held_stamps`, `_held_stamps`,
   `_Take.stamps` become `timestamps`, `_timestamps`, ... **(seen)**. Why: glossary, "A time stamp: `timestamp`".
   Check: `test_screen_recorder`.

## D. Text

D1. `record.state`'s description names ERROR too; `ScreenFrameStore`'s docstring drops "cleared by one thread" (the
   main thread clears it at begin and stop, the recorder's in `_free_store`) and its reason (pageable RAM), which
   VIDEO.md already gives **(seen)**. Why: "A docstring says what the thing is"; "Keep `Field` descriptions to one short
   line of panel text". Check: none, text only.

D2. VIDEO.md matches the code: at exit a look-back recording is not written (`stop` holds only a recording that is not
   look back); the sentence on the single 99.6 ms tick of an earlier run goes **(seen)**. Why: docs, a document
   describes what is built. Check: none, text only.

## Rules

- Use `Lock` and `Event` for shared mutable state and lifecycle signals
- A thread's loop catches an exception per item, logs it and goes on with the next
- A part that drops an item shows the count to the user
- Inside a package, code that does the same is written once
- Use `BaseSettings` for reactive configuration and `@dataclass` for plain value objects
- Avoid unnecessary copies
- Use no synonyms for one concept across modules; when a term changes, rename it everywhere in the same change
- A time stamp: `timestamp`; not `time`, `time_stamp`
- A docstring says what the thing is
- Keep `Field` descriptions to one short line of panel text: what the value is and its unit
- A document describes what is built; what is not built is in the plan
