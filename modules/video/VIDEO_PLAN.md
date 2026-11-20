# The video module's plan

## A. A clip's frame count

`StreamReader.frame_count` gives an mp4's stored frame count, but estimates an mkv's from its duration × fps, which
can read one short. The recording package counts each chunk's frames with it (`recording_folder.py`). Looked at when
the apps' plan F reaches the video module.

## E. Playback and the panel

- E3. Playback. A `play` switch on a held take: the recorder's thread sets the take's frames into `ScreenPreview` at their recorded times, looping between in and out, and the compositor draws it over the main window. Play is off while a save runs, as both use the recorder's thread. The secondary windows keep the last show picture, frozen. While it plays the show is held: each app reads `ScreenRecordingCompositor.playing`, a property, not the `play` setting, and skips the flow's work and draw, dropping camera frames as a frozen frame does, and skips the fluid's steps, so the fluid goes on where it stopped. The existing `pause`, which holds the camera frame while the fluid runs on with it, is renamed `freeze_frame` and keeps what it does. In deep_flow and low_flow: settings, renders and every preset. Check: both apps' tests; one `-sim` run, a held take played and looped, ticks steady, the show going on where it stopped. One commit.
- E4. The panel in four groups, in the order the code runs: Frame (main window, size, portrait, width, height, snap, fill), Quality (chroma, bit depth, frame step; max duration as readout), Record (start, stop, look back, duration, RAM reserve; recorded and dropped as readouts), Take (in and out, play, save, discard, name, quality). The size and button binds, and their unbinds, move to the group objects. One meaning per button: in look back Stop holds the take, as in a recording, and `save` only writes. Presets migrated in both apps. VIDEO.md: the four groups, and Stop holding the take in look back. Check: guard tests, both apps' tests, both presets load. One commit.

## The rules it touches

- Fix a problem where it is made: A
- A part that only needs the latest value reads it at its own tick: E3
- Remove obsolete pathways, presets in sync: E4
- When a term changes, rename it everywhere in the same change: E3, `pause` to `freeze_frame`
- A settings class mirrors the code, in the order it runs: E4
- The show owns only what it does with it: E3, each app holds its own show
- Settings connect the user and a component, not two components: E3, the apps read a property
- A document describes what is built: VIDEO.md in E3, E4
- One plan step checked by a run or test is one commit: E3, E4
