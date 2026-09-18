# White Space — Sound

What the sound synth in Max receives from the app, for whoever builds the Max patch. One body plays
two engines: the values that drive the light synth (`POSE_INSTRUMENT.md`, *The body*) go to Max
unchanged, from the same frames the light reads.

The sender is `OscSoundSender` (`inout/osc_sound_sender.py`), which extends `OscSound`
(`modules/inout/osc_sound.py`). Its settings are `inout.osc_sound_sender`.

## Conversion table

The values the light instrument plays with, and the OSC message that carries each to Max. The
names on the wire are older than the documents' terms; this table is the translation.

The measures (`POSE_INSTRUMENT.md`, *What each measure means*):

| In the light instrument | Pose feature       | OSC address                  | Index | On the wire                    | The light reads it as   |
|-------------------------|--------------------|------------------------------|-------|--------------------------------|-------------------------|
| left shoulder           | `Angles`           | `/pose/{id}/angle/rad`       | 0     | −π..π; 0 hanging, ±π up        | absolute over π, 0..1   |
| right shoulder          | `Angles`           | `/pose/{id}/angle/rad`       | 1     | −π..π; 0 hanging, ±π up        | absolute over π, 0..1   |
| left elbow              | `Angles`           | `/pose/{id}/angle/rad`       | 2     | −π..π; 0 straight, ±π folded   | absolute over π, 0..1   |
| right elbow             | `Angles`           | `/pose/{id}/angle/rad`       | 3     | −π..π; 0 straight, ±π folded   | absolute over π, 0..1   |
| leg deviation           | `LegDeviation`     | `/pose/{id}/angle/legs`      | –     | 0..1; 0 standing, 1 bent       | as sent                 |
| body bend               | `TorsoTilt`        | `/pose/{id}/angle/tilt`      | –     | −1..1; −1 left, 1 right        | as sent                 |
| distance                | `Distance`         | `/pose/{id}/distance`        | –     | 0..1; 0 near edge, 1 far edge  | unconnected             |
| shoulder symmetry       | `AngleSymmetry`    | `/pose/{id}/angle/sym`       | 0     | −1..1; left minus right        | unconnected             |
| elbow symmetry          | `AngleSymmetry`    | `/pose/{id}/angle/sym`       | 1     | −1..1; left minus right        | unconnected             |

The events and the place (`POSE_INSTRUMENT.md`, *Events*):

| In the light instrument | Pose feature       | OSC address                  | Index | On the wire                    | The light reads it as                       |
|-------------------------|--------------------|------------------------------|-------|--------------------------------|---------------------------------------------|
| sync                    | `Similarity`       | `/pose/{id}/similarity/pose` | other | 0..1 per live player           | the pair's mean, from `sync_threshold` to 1 |
| hit                     | `PlayheadOffset`   | `/pose/{id}/playhead/offset` | –     | −π..π; azimuth minus playhead  | the ticks closest to 0 (`PlayheadCrossing`) |
| presence                | the pose itself    | `/pose/{id}/active`          | –     | 1 present, 0 gone              | the presence envelope's gate                |
| where the person stands | `Azimuth`          | `/pose/{id}/azimuth`         | –     | radians                        | the centre of the window and the mask       |

The angles are calibrated before they are sent (`POSE_INSTRUMENT.md`, *The body*), so the two fixed
points are 0 and π for sound and light alike. They go out signed: the sign is the side of the body
the limb passes. The light takes the absolute over π (`PoseInstrument._measure`); Max does the same
to read an arm as the light reads it.

The distance is how far the person stands from the fixture within the tracked zone: 0 at
`track.rig.zone_min_radius`, 1 at `track.rig.zone_max_radius`, clamped. The tracker reads it from
where the feet meet the floor (`TRACKING.md`); it is NaN without a reading.

The similarity is a row: index `n` is this person against live player `n`, `max_players` wide; a
pair that does not exist is sent as 0.
What it measures and how it is smoothed is in `SIMILARITY.md`. The light takes the mean of both
directions of a pair and opens the window from `PI.window.sync_threshold` to 1, eased
(`PoseInstrument._set_reaches`); Max gets the row and shapes it itself.

## The wire

One OSC bundle per LERP frame, over UDP, to `ip_addresses`:`port`. The frames are the LERP-stage
poses as the ghoster emits them (`main.py`, `Ghoster.add_sound_callback`), so every value is
smoothed and interpolated as the light's is.

| Setting        | Preset `studio` | What it is                      |
|----------------|-----------------|---------------------------------|
| `ip_addresses` | 192.168.1.40    | the Max machine                 |
| `port`         | 8000            | the port Max listens on         |
| `stage`        | LERP            | the pipeline stage that is sent |

## Slots

Every person is a slot `{id}` in `/pose/{id}/...`. With the preset `studio` (`max_players` 6,
`num_virtual` 8):

| Slots | Who                                             |
|-------|-------------------------------------------------|
| 0–5   | live players, the tracker's ids (`TRACKING.md`) |
| 6     | the dummy (`POSE_INSTRUMENT.md`, *The dummy*)   |
| 7–14  | active ghosts; passive ghosts are never sent    |

A slot that empties sends its reset, every value 0 and `active` 0, for two bundles and then
nothing. On shutdown one bundle resets every slot and sends `/global/state` −1.

## The full lists

`/pose/{id}/angle/rad` and `/pose/{id}/angle/vel` carry nine values in the order of
`AngleLandmark`; `/pose/{id}/angle/sym` carries six in the order of `SymmetryElement`:

| Index | `angle/rad`, `angle/vel` | `angle/sym`                         |
|-------|--------------------------|-------------------------------------|
| 0     | left shoulder            | shoulder                            |
| 1     | right shoulder           | elbow                               |
| 2     | left elbow               | hip                                 |
| 3     | right elbow              | knee                                |
| 4     | left hip                 | arms: shoulder and elbow together   |
| 5     | right hip                | legs: hip and knee together         |
| 6     | left knee                |                                     |
| 7     | right knee               |                                     |
| 8     | head                     |                                     |

## The other per-person messages

Sent with every person; the light instrument plays with none of them.

| Address                         | Values        | Range      | What it is                                      |
|---------------------------------|---------------|------------|-------------------------------------------------|
| `/pose/{id}/playhead/fade`      | 1 float       | 0..1       | deprecated; 1, or a surplus ghost fading out    |
| `/pose/{id}/angle/vel`          | 9 floats      | radians/s  | the angles' velocities                          |
| `/pose/{id}/time/motion`        | 1 float       | count      | motion time: rises about 1 per second of motion |
| `/pose/{id}/time/age`           | 1 float       | seconds    | how long the person has been tracked            |
| `/pose/{id}/bbox`               | 4 floats      | `BBox`     | centre x, centre y, width, height               |
| `/pose/{id}/similarity/gate`    | `max_players` | 0..1       | the motion gate: both of the pair moving        |
| `/pose/{id}/similarity/motion`  | `max_players` | 0..1       | the same row as `similarity/pose`               |
| `/pose/{id}/similarity/leader`  | `max_players` | −1..1      | zeros: White Space makes no leader scores       |
| `/pose/{id}/similarity`         | `max_players` | 0..1       | deprecated; the same row as `similarity/pose`   |

## The global messages

Sent once per bundle.

| Address                   | Value   | Range    | What it is                                                   |
|---------------------------|---------|----------|--------------------------------------------------------------|
| `/global/state`           | int     | −1..10   | the show state (`STATES.md`); −1 in the shutdown blackout    |
| `/global/state/progress`  | float   | 0..1     | the state's progress (`STATES.md`)                           |
| `/global/progress`        | float   | 0..1     | the show's progress                                          |
| `/global/playhead`        | float   | −π..π    | the playhead as an azimuth (`CALIBRATION.md`, *Playhead*)    |
| `/global/motor`           | int     | mode     | the motor command's mode                                     |
| `/global/volume`          | float   | 0..1     | the operator's fader (`CALIBRATION.md`, *Sound (Max)*)       |
| `/global/speaker/offset`  | float   | radians  | where speaker 0 stands (`CALIBRATION.md`, *Sound (Max)*)     |

The playhead and the motor mode are 0 while no show runs. The volume and the speaker offset are
operator settings and are never zeroed.

## The return path

Max sends two levels back, received by `OscReceiver` (`inout.osc_sound_receiver`, port 8001 in the
preset `studio`): `/WS/idle/blue/left` and `/WS/idle/blue/right`, one 0..1 float each, to the
`beam_blue_sound` layer (`LAYERS.md`).

## Open

The wire stays as it is while the installation is being finished: it works, and the patch is built
on it. Once the installation is finished, the wire takes the documents' terms, in one change of the
app and the patch together:

| Now                               | Then                        | What the patch changes                      |
|-----------------------------------|-----------------------------|---------------------------------------------|
| `/pose/{id}/angle/rad` 0, 1       | `/pose/{id}/shoulder`       | the address, and the index: left 0, right 1 |
| `/pose/{id}/angle/rad` 2, 3       | `/pose/{id}/elbow`          | the address, and the index: left 0, right 1 |
| `/pose/{id}/angle/legs`           | `/pose/{id}/leg_deviation`  | the address                                 |
| `/pose/{id}/angle/tilt`           | `/pose/{id}/body_bend`      | the address                                 |
| `/pose/{id}/angle/sym`            | `/pose/{id}/symmetry`       | the address                                 |
| `/pose/{id}/similarity/pose`      | `/pose/{id}/similarity`     | the address                                 |

The values keep their ranges; the angles stay signed radians. `/pose/{id}/distance` already has
its term. With it:

- The messages that carry nothing of their own go: the deprecated `/pose/{id}/similarity` and
  `/pose/{id}/playhead/fade`, `/pose/{id}/similarity/motion`, `/pose/{id}/similarity/leader`
- The messages the patch does not read go: of `/pose/{id}/angle/rad` 4–8, `angle/vel`,
  `time/motion`, `time/age`, `bbox` and `similarity/gate`, which ones it reads
  **(not verified: the patch is not in this repository)**; the motion features are future sources
  of expression (`POSE_INSTRUMENT.md`, *Open*)
- The sender becomes White Space's own and no longer extends `OscSound`, which serves the other
  apps; whether its transport (the thread, the UDP client, the send interval) is shared from
  `modules/inout`
- Whether the code follows the documents' terms too: `TorsoTilt` for the body bend, `legs` for the
  leg deviation

As the wire is now:

- The reset of `/pose/{id}/angle/rad` and `/pose/{id}/angle/vel` sends 17 zeros, the live message
  9 values (`OscSound._add_inactive_frame_messages`); whether Max depends on the list length
- `/pose/{id}/similarity/motion` sends the plain similarity row although its comment says
  motion-gated (`SIMILARITY.md`, *Open*); which of the three similarity addresses Max listens to
