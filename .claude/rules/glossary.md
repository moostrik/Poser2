---
description: The one name for each concept that appears across modules and apps, and the synonyms it replaces
---
# Glossary

Each bullet names the term to use and, after "not", the synonyms it replaces. A term covers its forms in identifiers, files, settings, panel text, preset keys and documents.

## Shared (modules and apps)

- A component's settings object: `settings` / `self.settings`; not `config`, `_config`
- A settings class: `<ClassName>Settings`, after the class it configures
- The id of a tracked person: `track_id`; not `tracklet_id`, `pose_id`
- A time stamp: `timestamp`; not `time`, `time_stamp`
- Camera identifiers: `cam_id`, `num_cams`, `cam_tracklets`; not `camera_id`, `num_cameras`. Class names keep `Camera` (`CameraFrame`, `CameraTracklet`)
- The number of tracked people: `max_players`; not `num_players`, `max_poses`, `num_tracks`, "id slots"
- A person in panel text and docs: "player"; not "person", "participant"
- The pose frame: `Frame` / `FrameCallback`; every other frame type is qualified: `CameraFrame`, `CameraFrameCallback`, `FaceFrameCallback`, `LightFrame`, `LightFrameCallback`
- The sequencer's state on the board: `HasSequencerState`, `set_sequencer_state`, `get_sequencer_state`; not `HasSequence`, `set_sequence`
- Feature words in full: `Velocity`, `Symmetry`, `Confidence`, `Foreground`, `DenseFlow`, `MotionTime`, `Temperature`, `Compositor`, `Geometry`; not `Vel`, `Sym`, `Conf`, `Frg`, `DFlow`, `MTime`, `TEMP`, `Comp`, `Geom`
- Exponential moving average: `Ema`; not `EMA`
- American spelling: `center`, `visualization`, `color`; not `centre`, `visualisation`, `colour`

## Render

- `Layer`: owns its FBO or texture
- `Renderer`: draws into a target it is given and owns no texture
- `Compositor`: owns an FBO and draws other layers into it

## Inout and video

- A network address: `ip`, and `ip_out` for a receiver's reply address; not `ip_address`, `ip_addresses`, `ip_address_out`
- An on/off switch: `enabled`; not `active`
- IP validation: `is_valid_ip` in `net_probe`; not `validate_ip`, `is_valid_ip_address`
- The video encoder choice: `Encoder`; not `CoderType`, `CoderFormat`
- Hardware decoding: `hwaccel`; not `hwdecode`, `hw_acceleration`

## Apps

- The beam fixtures in white_space: "beam light"; not "lamp"
- Settings groups shared by the apps: `session` / `control`, `tracker`, `input_fps`; not `record` / `core`, `track`, `fps`
- An app's renderer: `<App>Render`
- The camera count at an app's root: `num_cams`, separate from `max_players`

## smile

- One of the two secondary windows that drive 24 screens of the sphere: `wall` (`sphere.WALLS`, `WallRender`, `walls`, "On walls"); not `player`, `PLAYERS`, `PlayerRender`, "player walls"
- The held grin player: `grin_player`; the clip player: `clip_player`
- The image generator: `generator` (`GeneratorMessage`, `GeneratorState`, settings group `generator`); not `diffuse`, `diffusion_callback`, `pipeline`, `GenMessage`. `DIFFUSION` stays as a state name
- The sequence's end effect: bloom / fade (`bloom_*`, `fade_*`, `BloomLayer`); not `destroy`, `destruct`
- The music track: `music_id`; the visitor's style: `style_id`; not `audio_id`, `movement_id`
- The laughter test: `LaughterTest`; not `ScreenTest`
