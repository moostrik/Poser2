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
- A time something is smoothed over (ms): `smooth` (the depth velocity's `smooth`, `catch_up` and `xy_smooth`); not `time`
- The z of the 3D flow: `depth_velocity` (`DepthVelocity`, `DepthVelocitySettings`); not `depth_track`. `depth` is the camera's depth
- A swap buffer's other buffer, the previous one: `back` (`SwapFbo.back`, `SwapTexture3D.back`, `SwapTexture2DArray.back`); not `back_texture`. A texture object is passed as itself, never as a `.texture` of it; a layer's `texture` is the layer's output
- American spelling: `center`, `visualization`, `color`; not `centre`, `visualisation`, `colour`

## Flow

- What a bridge filter makes from the motion, in widths/s as a velocity and in widths/s² as an acceleration: a "reading"; not "signal", "band", "component", "event"
- The six readings: `trail` (the motion, held and smoothed), `change` (the whole change of speed, Δv, signed), and its four parts `start`, `bow` (a limb arriving), `stop`, `wake` (a limb leaving)
- The bridge's filters, named by what they do: `lowpass` (`VelocityLowpass`, gives the trail) and `bandpass` (`VelocityBandpass`, `Change`, gives the change and its parts); not `VelocityTrailBridge`, `VelocityChangeBridge`, `impulse`, `MotionEvent`
- The pixels around one, which tell a part of the body changing speed from a limb arriving or leaving: `region` (`start_region_size`, `start_region_weight`, `stop_region_*`); not `patch`, `tile`, `gate`
- The two systems a fluid takes the bridge in: `boundary`, the body in the water, what the water is dragged toward (`boundary_velocity`, `boundary_density`, `boundary_temperature`); `body_force`, the Navier–Stokes force term, what is added × dt (`body_force_acceleration`, `body_force_density`, `body_force_temperature`); `wall`, the domain's edge
- "impulse", "body", "force", "delta" and "drag" name nothing on their own; a name says the quantity (`velocity`, `acceleration`) or the textbook term (`body_force`), or pairs with what it acts on (`boundary.velocity_drag`, the time the water takes to close in on the body's velocity)

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
- The gate on the group's similarity: "group sync" (`group_sync`, `GroupSync`, `GroupSyncGate`); not "full sync". The OSC address stays `/global/sync`
- The flow stages in an app's settings and `Layers`: `motion`, `bridge`, `fluid`; not `flow`, `fluid3d`, `fluid_flow`
- How far a field is carried along the water per step, 1 = with the water: `advection_scale` (on `velocity`, `density`, `temperature`); not `speed`, `self_advection`. `advection` is the method (semi-Lagrangian, MacCormack)
- The fluid's inputs sit in `input`, mirroring the bridge's outputs: `layer`, `spread` (the body's thickness in z, widths, its width at half its height), then `boundary` (`velocity_drag` and `velocity_gain`, `density_drag` and `density_gain`, `temperature_drag` and `temperature_gain`, a line per field: the time the water takes to follow the body, and the body's value times the gain as the target) and `body_force` (`acceleration_gain`, `density_rate`, `temperature_rate`, what a push adds per second); the dye's presence is 0 to 1, the speed in widths/s capped at 1, so a part moving at 1 width/s is fully there, density 1 is fully dyed water and warmth ±1 full; a field's `fade_time` and `evaporation_time` are seconds, 0 off; the box holds one `wall` for every field and side, and the look is `composite`, set in widths: `opacity` (the widths of fully dyed water that hide half of what is behind it), `fog` (the widths of water that dim light to half), `tint` (the water's color, white none) and `emission` (the glow of thick density, 1 its color); not `absorption`
- The 3D fluid's grid size: `width`, `height`, `depth` (its cells in z); not `layers`. The box's depth over its width: `depth_ratio`, the z cells stretched to it. Outside a stage's size, `depth` is the camera's depth. The fluid's z look and injection: `volume` (`VolumeSettings`)
- The 3D fluid's density carriers: the simulation density (in the simulation's cells, `SimulationDensity`, always on), the upscaled density (its amount at its own `width`, `height`, `depth`, `UpscaledDensity`, settings group `upscaled`) and the particles (`ParticleDensity`, group `particles`), each of the last two switched by `enabled`; not "grid density", "output density"

## smile

- One of the two secondary windows that drive 24 screens of the sphere: `wall` (`sphere.WALLS`, `WallRender`, `walls`, "On walls"); not `player`, `PLAYERS`, `PlayerRender`, "player walls"
- The held grin player: `grin_player`; the clip player: `clip_player`
- The image generator: `generator` (`GeneratorMessage`, `GeneratorState`, settings group `generator`); not `diffuse`, `diffusion_callback`, `pipeline`, `GenMessage`. `DIFFUSION` stays as a state name
- The sequence's end effect: bloom / fade (`bloom_*`, `fade_*`, `BloomLayer`); not `destroy`, `destruct`
- The music track: `music_id`; the visitor's style: `style_id`; not `audio_id`, `movement_id`
- The laughter test: `LaughterTest`; not `ScreenTest`
