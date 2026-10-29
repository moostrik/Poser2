# White Space — Layers

The light layers behind the states' mixes (see `STATES.md` for the choreography; this
document covers the layers themselves). All show layers are **indexed** below. For
`beam_playhead` and `beam_flash` the module docstrings are the source of
truth for behavior; `beam_blue_sound`, `flood` and `beam_wind_down` keep their full design sections
here; `pose_instrument`'s design (its voice, controls, patch and poses) is `POSE_INSTRUMENT.md`, and
its section here covers the layer alone.

Modes: **beam** layers write the four beam lights by name (`LightFrame.beam_lights`, indexed by
`BeamLightId`: front/back white, left/right blue) and no pixels; **projection** layers draw the
persistence-of-vision projection. The fixture's readout mode follows the *commanded* rpm — beam
mode below 200, projection mode at or above, switching on receipt of the rpm regardless of the
bar's actual speed (`loop` in `firmware.cpp` sets `SLOW` from `RPM`). The light sender maps the
beam lights to the firmware's pixel slots exactly when the fixture reads them
(`inout/osc_light_sender.py`), and the render simulates the beams in its own layer
(`render/layers/beam_light_simulation_layer.py`). Base classes `BeamLayer` / `ProjectionLayer` encode this, and the
folders and settings groups follow the same single axis (`layers/beam/` ↔ `light.beam_layers`,
`layers/projection/` ↔ `light.projection_layers` — no separate test folder). The file path names
the layer: the class is the folder and file in CamelCase and equals the `LayerId`
(`beam/flash.py` → `BeamFlash` → `beam_flash`; `projection/test_fill.py` → `TestFill` → `test_fill`;
`projection/` adds no prefix). Debug-only layers state that role in their docstrings; the
projection block's generic patterns keep a `test_` prefix.

## Inputs

Show layers and the state machine read pose frames from the board, never the tracker's tracklets. The
state machine also reads the board's hit streak (`HitSync`, `STATES.md` *Vocabulary*: in sync) and, as the pose
instrument does, the group sync on the board's LERP group record (`GroupSync`, the pipeline's gate; `STATES.md`
*Vocabulary*: group sync); the live `Similarity` feature stays the pipeline's and the sound's.
A player is present while their pose exists: the pose pipeline stops posing a player
`pose.tracklets.detection_timeout` (1.0 s) after their last detection, and every filter downstream
resets a track the moment its pose is missing. A layer adds no presence test of its own. A player's
azimuth is the pose's `Azimuth`, at their eyes: raw from CLEAN, smoothed at SMOOTH, predicted at
PREDICT and interpolated at LERP. The LERP stage runs inside the light tick at `light_rate`, one live root
field that the conductor's clock, the layers' playhead step (`Tick.interval`) and the LERP interpolators all
read, at the time the tick was paced for (`Tick.deadline`), so a LERP frame's stamp is regular whatever the tick's lateness. Show layers and the state machine read LERP poses, the stable eye azimuth and its `PlayheadOffset`. Their angles are the calibrator's (`POSE_INSTRUMENT.md`, *The
body*): 0 at neutral, π at raised. Their age is the pose's `Age`. The dummy (`POSE_INSTRUMENT.md`, *The dummy*)
joins the poses before the LERP filters at its own id, `max_players`, so to every layer it is a
player. The render draws every LERP pose as a figure over the
projection row at its azimuth (`render.pose_figures`). How the tracker produces the poses is in
`TRACKING.md`, *Downstream*.

The hit is detected once, by the LERP step `HitExtractor` (`pose/hit.py`, the one `PlayheadCrossing` on the
LERP `PlayheadOffset` and the step `beam_rpm` at the light rate), and stamped on the pose as `Hit`; `beam_flash`,
`pose_instrument` and the state machine read it there, so they agree by construction. `HitSync`, the LERP step
after it, records in its streak only the hits on poses at least `pose.hit_sync.min_age_seconds` old
(`STATES.md` *Vocabulary*: in sync); the hit itself, the flash and the instrument's mark fire on every crossing.

## Index — show layers

| Layer                 | Mode       | Reads                                      | Writes                                        | Used by                   |
|-----------------------|------------|--------------------------------------------|-----------------------------------------------|---------------------------|
| `beam_playhead`       | beam       | — (settings only)                          | front white beam light                        | S1–S6, S9–S11             |
| `beam_flash`          | beam       | LERP frames (PlayheadOffset)               | front white + blue beam lights; board flashes | S4                        |
| `pose_instrument`     | projection | LERP frames                                | white and blue lines, masks, marker           | S6 (projecting), S7, S8   |
| `flood`               | projection | — (settings only)                          | whole projection white                        | S8                        |
| `beam_wind_down`      | beam       | tick clock                                 | both white beam lights, fading                | S9, S10                   |
| `beam_sync_wind_down` | beam       | tick clock; `pose_instrument.sync_strobe`  | the wall strobing white ↔ blue, fading        | S11                       |
| `beam_blue_sound`     | beam       | sound levels from Max (board)              | left/right blue beam lights                   | S1, S2, S3, S5, S10       |

`beam_flash`'s blue beam lights are zeroed in the presets: S4 runs blue-none by design.

`beam_wind_down` is `flood`'s ending and a plain beam layer: the fixture is in beam mode from
S9's or S11's first packet, so the wall while the bar is still fast is the two white beam lights
spinning. See its section below.

Debug layers (never in a state's mix; reached via the `light.debug` select — choosing a
layer turns debug on: it shows solo at full weight and the motor follows its
mode, OFF returns the show): the beam tool `beam_test` (direct
levels for the four physical beam lights: front/back white, left/right blue — beam mode's
hardware check), the projection `projection_playhead` (the playhead's marker alone, with the
instrument's `pose_instrument.playhead` settings, for looking at the playhead by itself: `CALIBRATION.md`),
plus the projection `test_`-prefixed patterns: `test_pose_waves` (a wave/void
instrument, a reference/montage visual), `test_harmonic`,
`test_player_lines`, `test_calibration`, `test_fill`, `test_random`, and the waveform
patterns `test_pulse`, `test_chase`, `test_lines`, which draw the light synth's pulse with
`width` and `hardness` as in `LIGHT_SYNTH.md`.

---

## beam_blue_sound (BeamLayer)

The soundscape made visible: the left and right blue beam lights breathe with the actual
sound Max is playing.

- **Used by**: S2 IDLE and S3 IDLE_INTRO at full; S1 OFF_IDLE, S5 INTRO_IDLE and S10 END_IDLE fading in
- **Input**: `/WS/idle/blue/left` and `/WS/idle/blue/right` from Max — one float each,
  0..1, real OSC — received on the OSC sound receiver (`inout.osc_sound_receiver`) and
  stored on the board (sound-level store: levels + received-timestamp). Max sends these
  messages (`SOUND.md`).
- **Behavior**: left level → left blue beam light, right level → right blue beam light; a gain scales
  the mapping. Latency first: no softening — only a minimal smoothing window of at
  most 2–3 light frames (63–94 ms at the studio preset's 32 fps), there purely to bridge OSC-arrival vs light-tick
  timing jitter, never to shape the response (Max shapes the envelope; the beam lights follow).
- **Stale input**: when no message has arrived for `stale_seconds`, the layer falls back —
  either off, or a gentle idle pulse (tunable choice) — so a silent or disconnected Max
  never freezes the beam lights at a stuck level.
- **Settings**: `gain`, `smoothing_frames` (0–3, default 2), `stale_seconds`,
  fallback mode/level
- **Reset**: clears the smoothing window (beam lights re-attack from the live levels)

## flood (ProjectionLayer)

The whole projection constant white — the END's wall of light. The layer has no dynamics: the
cross-to-full is mix weights set by the states. The *ending* of the wall belongs to `beam_wind_down` — S8's flood
at 1.0 hands over to S9/S10's wind_down starting at the full wall, seamlessly.

- **Used by**: S8 END (easing in as the instrument fades)
- **Input**: none
- **Behavior**: `white[:] += level`; stateless
- **Settings**: `level` (white; blue stays 0 — the flood is a white statement)
- **Reset**: no-op

## beam_wind_down (BeamLayer)

The dying wall of light: owns the S9–S11 ending fade. It writes the two white beam lights at a
fading level; everything else is physics. The fixture is in beam mode from S9's or S11's first
packet (its readout mode follows the commanded rpm), so the two beam lights spin at whatever
speed the bar still has — a wall of white while fast, thinning into two beams as it
slows — and the fade rides through both. One mechanism; the layer never needs to know
when the bar is slow. The S8 → S9 hand-off is seamless at the DACs: `flood` at 1.0 in projection
mode drives the same two white outputs as this layer at 1.0 in beam mode.

Two layers are instances of it. `beam_sync_wind_down`, the **strobing wall** of S11, is built with
the instrument's `pose_instrument.sync_strobe`: the sync end leaves PLAY with the patterns strobing at the top
rung, and the wall keeps that strobe on the beam lights — on the strobes' shared tick grid
(`LIGHT_SYNTH.md`, *The strobe*) at 16 per second, a strobe frame swaps the wall to the two blue
beam lights (INVERT, as the patterns swap white and blue) or blanks it (BLACK). The first S11 frame
continues the last PLAY frame's strobe; only the picture under it changes. The amplitude is the
fade, so the strobe dies with the wall into the dim line underneath. The wall carries no
`frame.sync`: `/global/sync` is the instrument's and reads 0 under it (`SOUND.md`).

- **Used by**: S9/S10 as `beam_wind_down`, S11 as `beam_sync_wind_down`, at constant weight 1.0
  (reset on state entry). The states put the landing look underneath (`beam_playhead` at
  DIM/BRIGHT, `beam_blue_sound`) — it is revealed as the wall dies, so nothing has to splice or
  match at the hand-off.
- **Input**: the tick clock; no pose data, no playhead signals. The strobing wall reads
  `pose_instrument.sync_strobe.mode`.
- **Behavior**: both white beam lights at `f × level`, `f = 1 − ease(elapsed / spin_down_seconds)`,
  hand-tuned to ride the physical spin-down (the sensor is silent above 200 rpm, so the
  deceleration is not measurable). The states exit once `progress` reaches 1 and the
  playhead lock holds (`STATES.md`). The fade advances only while the layer is drawn, so a
  debug solo pauses it.
- **Settings**: `level`, `spin_down_seconds` (hidden — the visible slider is
  `states.spin_down_seconds`, next to `spin_up_seconds`, shared in via the root),
  `progress` (read-only fade readout 0..1: the states' exit, `/global/state/progress` and S10's
  sound-visual reveal ride it)
- **Reset**: restarts the fade at the full wall (called from S9–S11 `enter()`)

## pose_instrument (ProjectionLayer)

The heart of the piece: each player stands in a dim blue mask at their azimuth, and around them
lies a pattern of full white and full blue lines drawn from their pose. The bridge (its
measures, connections, events, how people combine, the mask and the hit) is `POSE_INSTRUMENT.md`,
and the light synth that draws the lines is `LIGHT_SYNTH.md`.

- **Used by**: S6 (once projecting), S7, S8
- **Input**: from the LERP frames, `Azimuth`, `Angles`, `LegDeviation`, `TorsoTilt`, `Distance`,
  `AngleSymmetry`, `Similarity` and `PlayheadOffset` (`POSE_INSTRUMENT.md`, *The body*). The
  layer adds no smoothing.
- **Settings**: the layer's own group holds only `blend`; everything else is the root `pose_instrument` group
  (`POSE_INSTRUMENT.md`, *Settings*), among them `opposite`, which draws the patterns half a turn
  from their people while the masks stay on them
- **Strobe**: each oscillator's lines can be gated whole, on or off per tick, on the clock's shared
  tick grid (`LIGHT_SYNTH.md`, *The strobe*); a dark tick is one dark revolution because the frame
  rate equals the revolution rate (32 fps and 1920 rpm in the studio preset)
- **Reset**: forgets every player and pass (S6's entry, a fresh instrument per cycle)

---

## Open

- **beam_blue_sound**: fallback choice (off vs idle pulse); exact `/WS/idle/blue/left` and `/right` scaling agreed
  with Max (linear 0..1 vs dB)
- **pose_instrument**: see `POSE_INSTRUMENT.md`, *Open*
- **The playhead boundary**: `PlayheadOffset` and `Hit` are pose features stamped at LERP, and the hit streak
  a group feature there: the playhead is the one light datum the app's pose steps read (`playhead_phase`,
  `playhead_bars` off the board, `beam_rpm` and the light rate from the settings). They are White Space's own
  (`apps/white_space/pose`), so `modules/pose` knows no playhead; the light layers and the state machine read the
  results off the frames and the record, never detect them (roadmap step 6, light to pose)
