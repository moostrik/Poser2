# White Space — Pose Instrument

The instrument the participants play with their bodies: the light half of pose → sound. This
document is the instrument's design: its voice, its controls, the poses that mean something, the
patch that connects them, and the alternatives the patch can be re-routed to. The layer that draws
it is indexed in `LAYERS.md`; the show states that run it are in `STATES.md`.

## The voice

Each person stands in a dim blue **band**, their root. Around it, mirrored about them, their
**voice**: a pattern of lines, full white and full blue, cut off at the **reach** each side. Two
oscillators make the voice, one per colour, both spatial: they oscillate over the distance from the
person, not over time. Nothing in the projection moves by itself. A line moves only when the pose
that draws it moves.

| LFO term      | In the instrument                   | Setting            |
|---------------|-------------------------------------|--------------------|
| waveform      | cosine, thresholded into lines      | `LinePattern`      |
| rate          | the **interval** between lines      | `interval`         |
| pulse width   | the **duty**: line thickness        | `duty`             |
| phase         | the lines' offset from the band     | `phase`            |
| overtone      | a second oscillator at `harmonic_order` × the rate, mixed in | `harmonic`, `harmonic_phase` |
| envelope      | reach grows on arrival, shrinks on leaving | `attack_seconds`, `release_seconds` |
| accent        | the playhead's crossing widens every line one tick | `hit_widen` |
| unison        | sync: two voices' reach grows until they overlap  | `sync_threshold`  |

The waveform per colour, `x` the distance from the person in pixels:

```
u   = x / interval + phase
lfo = (1 − harmonic) · cos 2πu + harmonic · cos 2π(order · u + harmonic_phase)
lit = lfo ≥ cos(π · duty)
```

Duty 0 is silence (no lines), duty 1 a held note (solid), duty ½ a square wave (lines and gaps
equal). White and blue are separate projections on the fixture **(site fact)**, so the two voices
sound four tones: dark, blue, white, and white over blue. With equal intervals and phases half an
interval apart the voices are complementary, alternating white and blue; with the phases equal they
stack into the overlap tone with dark between; with unequal intervals they beat.

### Constraints

| Constraint                                          | In the instrument                                  |
|-----------------------------------------------------|----------------------------------------------------|
| ~90 line/gap pairs per turn resolve **(site fact)** | `min_feature` 2°: no line or gap narrower           |
| Brightness does not read **(site fact)**            | every pixel 0 or 1 per colour; only the band is dim |
| No jitter                                           | lines follow the current pose and nothing else      |
| People recognise their own voice                    | each pattern mirrored about its person              |
| Sync makes more visible, never thicker              | reach grows; the lines stay as they are             |

The interval never goes below twice `min_feature`, and a line and its gap are each at least
`min_feature`; below that a voice falls silent, above `interval − min_feature` it holds. A band or the
reach edge cuts a line where it falls: lines slide out from behind the band and into view at the
reach. Two overlapping voices union; a gap narrower than `min_feature` fills.

## The controls

Six pose values come from the LERP frames, already smoothed by the pose pipeline. Each becomes a
control in 0..1. The output is mirror-symmetric, so a control has to be symmetric too: which arm is
which cannot show as left and right in the light, only as a different sound.

### One joint at a time

| Joint          | Feature                    | Range          | Alone it is                                              |
|----------------|----------------------------|----------------|----------------------------------------------------------|
| left shoulder  | `Angles.left_shoulder`     | 0 hanging, π up | half of the lift, or one voice's fader (*Symmetry*)     |
| right shoulder | `Angles.right_shoulder`    | 0 hanging, π up | the other half, or the other voice's fader              |
| left elbow     | `Angles.left_elbow`        | 0 straight, π folded | half of the overtone mix                            |
| right elbow    | `Angles.right_elbow`       | 0 straight, π folded | the other half                                      |
| legs           | `LegDeviation`             | 0 standing, 1 bent | **deviation**: how far the rate is pushed             |
| torso          | `TorsoTilt`                | −1 left, 0 upright, 1 right | **pitch bend**: signed, returns to centre    |

### As groups

| Control      | From the joints                                | Music                                             |
|--------------|------------------------------------------------|---------------------------------------------------|
| `LIFT`       | mean of both shoulders / π                     | the fader between the two voices: blue up to white |
| `ARM_SPLIT`  | \|left − right\| shoulder / π                  | one arm up: a solo line against the other voice    |
| `BEND`       | mean of both elbows / π                        | overtones: the note gains harmonics                |
| `BEND_SPLIT` | \|left − right\| elbow / π                     | where the overtone sits: on the beat or off it     |
| `LEGS`       | `LegDeviation`                                 | deviation of the rate: detune                      |
| `TILT`       | (`TorsoTilt` + 1) / 2                          | pitch bend, 0.5 at rest                            |
| `CONSTANT`   | 1                                              | a fixed knob                                       |

Arms first: the shoulders carry the loudest parameters, duty and interval. Elbows shape the timbre.
Legs and torso are the secondary players: deviation and bend.

### Symmetry: what the two arms can do

The light is mirrored, so the two arms cannot be left and right. Four ways to give them two jobs:

| Way                     | Left arm, right arm                      | Mirror poses      | One arm up alone              |
|-------------------------|------------------------------------------|-------------------|-------------------------------|
| sum and difference (now)| `LIFT` = mean, `ARM_SPLIT` = \|difference\| | sound the same | half lift, full split         |
| each arm a voice        | left = white fader, right = blue fader   | swap the colours  | left: white over blue; right: dark |
| lead and accompaniment  | high arm = lead, low arm = accompaniment | sound the same    | the lead alone                |
| signed difference       | left − right as a bipolar control        | mirror the bend   | bends one way or the other    |

"Each arm a voice" gives the richest arm vocabulary: every arm combination is a different chord of
the four tones (down/down blue, up/up white, left up the overlap tone, right up dark), and the
elbows can follow the same split (left elbow the white overtone, right elbow the blue). It costs the
mirror identity: a person and their mirror image sound different. The same four ways apply to the
elbows. The `PoseControl` list carries the first way; the others are additions to it.

## The patch

A patch routes one control into one parameter: `parameter = low + (high − low) × control^curve`.
Per colour there are five patches; the panel re-routes them live and the preset stores them.

### Now (`studio.json`)

| Colour | `duty`     | `interval`           | `harmonic`   | `harmonic_phase`   | `phase`         |
|--------|------------|----------------------|--------------|--------------------|-----------------|
| white  | LIFT 0 → 1 | LEGS 0.5 → 0         | BEND 0 → 1   | BEND_SPLIT 0 → 0.5 | TILT 0 → 1      |
| blue   | LIFT 1 → 0 | ARM_SPLIT 0.5 → 0.15 | BEND 0 → 0.6 | CONSTANT 0.5       | TILT 0.5 → −0.5 |

Intervals span `interval_min` 4° to `interval_max` 24° (patch value 0 → 1), `harmonic_order` 2.

| Pose value      | Connection now                                        | Alternatives                                                  |
|-----------------|-------------------------------------------------------|---------------------------------------------------------------|
| shoulders, both | the fader: white duty up, blue duty down              | each arm a voice; lift also opens the interval (a swell)      |
| shoulders, split| blue's rate: one arm up tightens the blue lines       | a solo overtone on one voice; the signed split as a bend      |
| elbows, both    | overtones on both voices, more on white               | overtones on one voice only; elbows as the interval instead   |
| elbows, split   | where the white overtone sits (on → off the beat)     | the overtone's mix on one voice; elbow split as blue's phase  |
| legs            | deviation: bent legs push the white rate up (denser)  | detune blue from white only, so the voices beat; deviation as duty depth |
| torso           | opposite phase shifts: lean changes the overlap tone  | pitch bend: lean bends both rates up or down, signed          |

Deviation is the legs' word: how far the rate is pushed from its rest. The rest rate is the
standing pose; a bent knee or a stretched hip pushes it, and the further the legs deviate the further
the rate. Pitch bend is the torso's: signed, and it comes back to centre when the person stands
straight. Both are alternatives to the phase routing now, on the same knobs.

## Poses that hold meaning

The arm combinations name themselves after what they sound like. Angles are the shoulder and elbow
angles the pose pipeline measures; the controls follow from *As groups*.

| Pose            | Body                                              | Controls                    | Voice (patch now)                                   |
|-----------------|---------------------------------------------------|-----------------------------|-----------------------------------------------------|
| **Drone**       | arms hanging, standing straight                   | all 0                       | solid blue over the reach: the pad                  |
| **Pulse**       | arms out level (a T)                              | LIFT ½                      | square wave: white and blue lines alternate, equal  |
| **Tutti**       | both arms straight up (a V)                       | LIFT 1                      | solid white: everyone in                            |
| **Call**        | one arm up, one hanging                           | LIFT ½, ARM_SPLIT 1         | white pulse against tight blue lines: a solo        |
| **Chord**       | hands on hips, elbows folded, shoulders low       | BEND 1                      | the drone splits into overtone lines                |
| **Cluster**     | hands on the head, arms up and folded             | LIFT ~½, BEND 1             | dense split lines in both voices                    |
| **Syncopation** | one arm folded, one straight, both level          | LIFT ½, BEND ½, BEND_SPLIT 1| the white overtone sits off the beat                |
| **Bend**        | any of the above, leaning                         | TILT ≠ ½                    | the two voices slide across each other: the overlap tone comes and goes |
| **Detune**      | any of the above, in a lunge or a crouch          | LEGS > 0                    | the white rate rises: the voices beat               |
| **Accent**      | any pose, on the playhead's crossing              | hit                         | every line widens for one tick                      |
| **Unison**      | two people holding the same pose                  | similarity ≥ `sync_threshold` | the reaches grow until the two voices overlap     |

Every combination of the six values draws differently: all six controls are routed, and no two poses
in the table share a control set.

## Tuning

The patch, the ranges and the levels are settings: live from the panel, saved in the preset. The
drawing itself (`PoseInstrument`, `LinePattern`) hot-reloads on save. Adding a control or a
parameter needs a restart.

## Open

- The patch: which routing plays best on the machine (the composition work)
- Whether the arms should each be a voice (mirror images sound different) or stay sum and difference
- Whether the torso should bend the pitch (interval) or shift the phase as now
- Whether the legs should detune blue from white (beating within one voice) or push both rates
- A parameter takes one source; pitch bend and deviation on the same interval need a second patch
  summed into it
- Whether a line appearing at `min_feature` wide flickers when a pose hovers at its threshold
  **(deduction: the LERP smoothing should hold it)**
- Whether `min_feature` 2° holds with the moiré of two synced voices on the machine
- Camera level: a pitched wide-FOV camera reads a small spurious tilt near the frame edges, a
  pitch bend nobody played; check a straight person at the edge
