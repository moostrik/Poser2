---
description: The one name for each thing that more than one package uses, and the names it had before
---
# Glossary

Names for things that more than one package uses. Each bullet gives the name and, after "not", the names that thing had before. A word after "not" is free for anything else.

## Modules and apps

- A component's settings object: `settings` / `self.settings`; not `config`, `_config`
- A settings class: `<ClassName>Settings`, after the class it configures
- A root group's named file: "settings file"; not "preset"
- The id of a tracked person: `track_id`; not `tracklet_id`, `pose_id`
- A time stamp: `timestamp`; not `time`, `time_stamp`
- Camera identifiers: `cam_id`, `num_cams`, `cam_tracklets`; not `camera_id`, `num_cameras`. Class names keep `Camera` (`CameraFrame`, `CameraTracklet`)
- The number of tracked people: `max_players`; not `num_players`, `max_poses`, `num_tracks`, "id slots"
- A person in panel text and docs: "player"; not "person", "participant"
- The pose frame: `Frame` / `FrameCallback`; every other frame type is qualified: `CameraFrame`, `CameraFrameCallback`, `FaceFrameCallback`, `LightFrame`, `LightFrameCallback`
- The sequencer's state on the board: `HasSequencerState`, `set_sequencer_state`, `get_sequencer_state`; not `HasSequence`, `set_sequence`
- Feature words in full: `Velocity`, `Symmetry`, `Confidence`, `Foreground`, `DenseFlow`, `MotionTime`, `Temperature`, `Compositor`, `Geometry`; not `Vel`, `Sym`, `Conf`, `Frg`, `DFlow`, `MTime`, `TEMP`, `Comp`, `Geom`
- Exponential moving average: `Ema`; not `EMA`
- An on/off switch: `enabled`; not `active`
- A layer that owns no texture and draws into a target it is given: `Draw`; not `Renderer`
- American spelling: `center`, `visualization`, `color`; not `centre`, `visualisation`, `colour`

## Apps

- Settings groups shared by the apps: `setup`, `session` / `control`, `tracker`, `input_fps`; not `record` / `core`, `track`, `fps`
- An app's renderer: `<App>Render`
- The camera count at an app's root: `num_cams`, separate from `max_players`
- The gate on the group's similarity: "group sync" (`group_sync`, `GroupSync`, `GroupSyncGate`); not "full sync". The OSC address stays `/global/sync`
