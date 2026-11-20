# The video module's plan

## A. A clip's frame count

`StreamReader.frame_count` gives an mp4's stored frame count, but estimates an mkv's from its duration × fps, which
can read one short. The recording package counts each chunk's frames with it (`recording_folder.py`). Looked at when
the apps' plan F reaches the video module.

## E. The panel

- E4. The panel in four groups, in the order the code runs: Frame (main window, size, portrait, width, height, snap, fill), Quality (chroma, bit depth, frame step; max duration as readout), Record (start, stop, look back, duration, RAM reserve; recorded and dropped as readouts), Take (in and out, play, save, discard, name, quality). The size and button binds, and their unbinds, move to the group objects. One meaning per button: in look back Stop holds the take, as in a recording, and `save` only writes. Presets migrated in both apps. VIDEO.md: the four groups, and Stop holding the take in look back. Check: guard tests, both apps' tests, both presets load. One commit.

## The rules it touches

- Fix a problem where it is made: A
- Remove obsolete pathways, presets in sync: E4
- A settings class mirrors the code, in the order it runs: E4
- A document describes what is built: VIDEO.md in E4
- One plan step checked by a run or test is one commit: E4
