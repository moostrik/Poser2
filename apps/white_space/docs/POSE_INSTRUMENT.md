# White Space — Pose Instrument

The instrument the participants play with their bodies. An instrument is what a player plays: a
keyboard synth is its keyboard, its synth engine and the patch that ties them. Here a person's
body takes the place of the keyboard.

| Part                           | In a keyboard synth      | Here                                                       |
|--------------------------------|--------------------------|------------------------------------------------------------|
| what the player acts on        | keyboard, wheels, pedals | the body, as the pose pipeline measures it (*The body*)    |
| what makes the result          | the synth engine         | the light synth (`LIGHT_SYNTH.md`)                         |
| what ties them                 | the patch                | which measure goes into which parameter, and by how much   |
| the whole                      | the instrument           | the pose instrument                                        |

One body plays two engines: the same measures drive the sound synth in Max and the light synth.
This document is the light's. It is the bridge between the pose data and the light synth, in four
parts. **Vocabulary** fixes the words. **Meaning** is what each measure of the body should say.
**Connections** is how the measures and the events are wired to the synth so that the meaning
comes out. **Implementation** is the bridge's design and the layer.
Meaning and connections are one loop: meaning sets the targets, the targets pick the connections,
the connections have consequences, and the consequences are new meaning to judge. While one is
held still the other is worked on.

The layer's place among the other layers is in `LAYERS.md`; the states that run it are in
`STATES.md`.

## The instrument

Each person makes light in the projection. At their azimuth sits a dim blue **mask**: it
goes over everything at the person, the patterns of everyone and the light of the other layers;
the playhead passes through it dimmed by a setting. Around the mask, mirrored about the person
(or, per oscillator, passing behind them), their **pattern**: lines of full white and full blue,
generated behind the mask and coming out from under it. Only a **window** of the pattern is
visible each side of the person; sync opens the window wider.

The ground rules, as set:

- About 90 lines per revolution, line and gap equal, is the visual maximum **(site fact)**. It is a
  guide: a good reason may go past it. It bounds the pitch and not the width of a line or a gap
  (`LIGHT_SYNTH.md`, *The rules*).
- Full on: every pixel of white and of blue is off or full, since the fixture projects on whatever
  is around it and subtle brightness is lost (`LIGHT_SYNTH.md`, *A light synth*). The mask is the
  one dim thing.
- No blinding: the instrument never blinds a person with white; people must be able to see each
  other. The mask is how, so full white can fill a window and never fall on a face. This rule is
  the instrument's alone, not the other layers'.
- White and blue are projected separately by the fixture **(site fact)**, so where both are on the
  overlap reads as a subtly different tone. The palette is four tones: dark, blue, white, both.
- No jitter: the pose pipeline's smoothing is the only smoothing. The instrument adds no smoothing
  and no steps, since steps are jitter of another kind, and no flicker of its own: a smooth change
  of the body is a smooth change on the projection.
- Lines follow the pose. On top of that they travel by themselves, slowly and peacefully, blue
  inward and white outward: a low speed that may be 0.
- The output each side of a person is symmetric while its oscillators mirror (the default),
  except the window's reach: sync opens one side.
- The pattern is the light synth's; the six measures play it; every combination of the six draws
  differently; the arms come first, legs and body bend second.
- Arms down is neutral: full blue over the window. Arms up: full white. In between, the pattern.
  Full is solid over the window, thinning at its ends (`LIGHT_SYNTH.md`, *The window*).
- Blue and white can behave differently; line thickness and interval can change.
- A hit (the playhead crossing the person) marks that person. Neither wider lines nor a tint of
  the lines reads as a mark **(site facts)**; the mark is the push and the mask's flash
  (*Events*), with the playhead's marker passing the person; each mark is optional through its
  levels.
- Sync makes more of the pattern visible: the window opens, the lines stay what they are.
- A person's pattern can be drawn half a turn from them (`opposite`), their mask staying on them:
  a switch for looking at the two apart, not for the show.

---

# Part 1 — Vocabulary

The language of the pattern is the light synth's and is in `LIGHT_SYNTH.md` (*Terms and what
belongs where*): oscillator, parameter, slot, source, output, pitch, interval, pulse width, phase,
speed, hardness, LFO, envelope, window, reach, taper, push, voice. Standard synth terminology
where the synth is a synth; our own word where the wall differs. The words of the bridge:

| Word       | What it is                                                                            |
|------------|---------------------------------------------------------------------------------------|
| measure    | a value of the body, measured by the pose pipeline (*The body*); ours: the body is the controller |
| source     | what is plugged into a parameter's slot: a measure, an LFO's or an envelope's output  |
| connection | a measure as the source in a slot: the measure, the parameter, the base, the amount   |
| patch      | everything set for the synth, the connections among it; shared by all voices          |
| voice      | one person's instance of the light synth                                              |
| event      | what the instrument triggers in a voice: presence, the hit, sync (*Events*)           |

What is the instrument's and what is the synth's (`LIGHT_SYNTH.md` states the same split from the
synth's side): the instrument decides what each source is, which measure, and the shaping of that
measure before it is handed over (the absolute of a travel, a remap; the dead zones are the
pipeline's, *The body*); when the gates open (presence, the hit);
the reaches; which output is white and which blue; and the mask. A source may also move in time
by the bridge's own hand, as the mask's flash does: the breath, a sine per person that swings a
width (*The connections*). Everything from the slot on is the synth's, and the synth never
changes a source before the slot and never adds two.

---

# Part 2 — Meaning

## What we hold on to

Two meanings are fixed, the ground rules' neutral and raised arms: the blue ping and the bass, as
the sound has them (`STATES.md`, S4: a glass ping for neutral, a heavy bass for arms raised).
Between the two the magic happens, and this part describes it: a pattern for every combination of
the measures, so each measure of the body has to mean something, and nothing but the two fixed
points is tied to a pose.

## The rules of meaning

1. Only measures mean something: the six measured values of the body, the distance, and what the
   pose pipeline derives from them (the symmetry of a pair). The instrument shapes one measure
   into one source (a dead zone, a remap); a value computed from several measures belongs in the
   pipeline.
2. A meaning belongs to a measure, never to a pose. The table says what happens when a measure
   changes. No pose owns a meaning; the two fixed points are the only exceptions.
3. Groups of measures (both arms, both elbows) and poses are consequences of the measures'
   meanings. They are read back in Part 3, never written here.
4. Raising one arm never means the same as raising the other, and the two arms are connected by a
   relation from the vocabulary of the light synth: not by a colour, and not by whatever two
   parameters were free.
5. The same holds for every pair of joints: a pair acts through a musical relation, or through its
   symmetry.
6. A meaning is spoken in the vocabulary: it names a term of the light synth.
7. Every combination of the measures draws differently.
8. Few connections, much meaning. A measure may mean nothing on its own; then its symmetry may.
   Not every parameter of the synth needs a measure.
9. Primary measures make the note, secondary measures modulate it. A secondary meaning needs a
   note to act on, so it is judged on a pattern, never on the fixed points.

The rules are aims, not gates. Two things are fixed: arms hanging is full blue and arms raised is
full white. Every combination drawing differently (rule 7) is the aim in the end, not a test of
each step. A calculation on several measures may live in the instrument while it is tried; once
the result is liked it moves into the pose pipeline (rule 1): the shoulders' mean and difference
are such calculations. The elbows depart from rule 4 as an experiment: each elbow plays its own
colour's pitch.

## What each measure means

Each measure is a musical term of the light synth: a parameter of an oscillator (pitch, pulse
width, phase, speed), the level of an LFO, a reach. **Primary** measures make the note: the arms,
worked out in `MATRIX.md`. **Secondary** measures modulate a note that is already sounding: the
legs and the body bend. The distance, how far the person stands from the fixture, is the one
measure that is not of the body's pose.

| Measure             | Role      | Term                                                            |
|---------------------|-----------|-----------------------------------------------------------------|
| shoulders (mean)    | primary   | the balance of white and blue: both pulse widths                |
| shoulder excess     | primary   | which arm is higher: its own colour's width breathes            |
| left elbow          | primary   | the white: its fold the pitch, its turn the flow                |
| right elbow         | primary   | the blue: its fold the pitch, its turn the flow                 |
| leg deviation       | secondary | open                                                            |
| body bend           | secondary | the flow: one way the white faster and the blue slower          |
| distance            | secondary | open                                                            |
| elbow symmetry      | secondary | open: it shows unconnected, as the colours in or out of tune    |

Two events sit over every measure: the **accent** of a hit and the **unison** of sync, the windows
of two people opening toward each other until the two patterns overlap and become one (*Events*).

## Composition

The aim is a balanced composition that people recognise they influence. Not every parameter of
the synth has to be used; they are there to be tried, because several may mean the same thing to
the eye, or the pattern's own workings may already produce what a parameter would add. A
connection earns its place when a person can find it with their body. Few connections, much
meaning: a measure may drive more than one parameter, and a parameter may stay at its base.

---

# Part 3 — Connections

## Pose results

The poses we check against: each with the light it should produce under the meanings of Part 2.
The list is the acceptance set, small on purpose; on the machine each row is a thing to stand in
and look at, and a test. The rows are the dummy's saved poses (`data/poses.json`, *The dummy*),
so each is one pick in the panel. The first two are the calibrator's reference poses, `neutral`
(arms hanging, standing) and `raised` (arms up); the others are built on them, the T halfway along
the calibrator's shoulder arc from neutral to raised. Every pose between the rows is unique and is
not described.

| Pose                                        | Result                                                           |
|---------------------------------------------|------------------------------------------------------------------|
| neutral                                     | full blue over the window: the blue ping                         |
| raised                                      | full white over the window, no blue: the bass                    |
| arms out level, a T                         | white and blue lines, each half the interval, touching           |
| left arm up, right hanging                  | the T's widths, the white breathing, the blue still              |
| right arm up, left hanging                  | the T's widths, the blue breathing, the white still              |
| a T, both elbows folded                     | the T's lines, finer, white and blue still in tune               |
| a T, left elbow folded                      | the white lines finer than the blue: the colours slide apart     |
| a T, right elbow folded                     | the blue lines finer than the white: the colours slide apart     |
| a T, leaning left                           | the T, the white slower and the blue faster                      |
| a T, leaning right                          | the T, the white faster and the blue slower                      |
| a T, in a crouch                            | the T: the legs play nothing                                     |
| \|__ (a T, the right forearm turned 90°)    | the blue finer and flowing against the white: a moving moiré     |
| \_\_\| (a T, the left forearm turned 90°)   | the white finer and flowing against the blue: a moving moiré     |

## The body

The instrument's measures are pose features and nothing else: values the pose pipeline measures
and publishes with every frame, smoothed. The pipeline derives the leg deviation and the symmetry
of a pair from the joint angles. The distance is the tracker's, read from where the feet meet the
floor (`TRACKING.md`, *The pose's distance*).

| Feature           | Measure                                                  | In the pipeline       |
|-------------------|----------------------------------------------------------|-----------------------|
| left shoulder     | 0 hanging → ±1 straight up, the sign the side it passes  | `ArmTravel`           |
| right shoulder    | 0 hanging → ±1 straight up, the sign the side it passes  | `ArmTravel`           |
| left elbow        | 0 straight → ±1 folded; its turn 0 straight → ±π folded  | `ArmTravel`, `Angles` |
| right elbow       | 0 straight → ±1 folded; its turn 0 straight → ±π folded  | `ArmTravel`, `Angles` |
| leg deviation     | 0 standing straight → 1 a leg fully bent                 | `LegDeviation`        |
| body bend         | −1 left → 0 upright → 1 right                            | `TorsoTilt`           |
| distance          | 0 the zone's near edge → 1 its far edge                  | `Distance`            |
| symmetry, a pair  | signed, left minus right: how unequal the two sides are  | `AngleSymmetry`       |

The angles are calibrated from two reference poses, arms hanging and arms raised
(`AngleCalibrator` in `modules/pose/nodes/filters`, settings `pose.angle_calibrator`), so the
fixed points are 0 and π for sound and light alike. The sign of an angle is the side of the body
the limb passes and means nothing to the instrument: it takes the absolute, so an arm raised
outward and one raised across the body read alike.

Every measure comes through a **dead zone** in the pipeline: it reads 0 up to its neutral zone, 1
from its far zone on, and linear between; continuous, so never a jump. A hanging arm reads a few
degrees and a raised one stops short of π, so without the zones the two fixed points are never
quite reached and the pattern never quite rests. The arms' zones make a feature of their own, the
**travel** (`ArmTravel`, from `ArmTravelExtractor` in `modules/pose/nodes/extractors`): where each
arm joint is along its travel from neutral to the far pose, the calibrated angle's magnitude taken
from the band between the two zones onto 0..1 with the sign kept. The angles themselves stay raw,
so the similarity, the symmetry, the velocity and the elbow's turn read the angles and only the
instruments read the travel, the light and the sound alike (`SOUND.md`). The body bend's and the
leg deviation's zones are in their extractors.

| Measure       | Dead zone                                  | Settings, under `pose`                                                          |
|---------------|--------------------------------------------|---------------------------------------------------------------------------------|
| shoulders     | around hanging and around raised, degrees  | `arm_travel_extractor.shoulder_neutral_dead_zone`, `shoulder_raised_dead_zone`  |
| elbows        | around straight and around folded, degrees | `arm_travel_extractor.elbow_neutral_dead_zone`, `elbow_folded_dead_zone`        |
| body bend     | around upright, degrees; full at the tilt  | `torso_tilt_extractor.neutral_dead_zone`, `tilt_degrees`                        |
| leg deviation | at standing, degrees; full at each top     | `leg_deviation_extractor.neutral_dead_zone`, `hip_max_degrees`, `knee_max_degrees` |

The distance and the symmetries get theirs with their connection; the similarity has its own
remap, `window.sync_threshold`.

## Symmetry

The light is mirrored, so left and right in the body never show as left and right in the light;
which arm is which shows only as a different sound. The **symmetry** feature (`AngleSymmetry`)
holds the meaningful pairs, the four joints (shoulder, elbow, hip, knee) and the arm and the leg as
a whole per side (`arms`: shoulder and elbow; `legs`: hip and knee, weighted as the leg deviation
is), each signed, left minus right, −1..1, so that downstream the signed value or its absolute can
be used. A person and their mirror image draw differently. `AngleSymExtractor` makes it; it is
windowed, graphed and sent to Max like the leg deviation.

## The connections

A **connection** is one measure as the source in one parameter's slot (`LIGHT_SYNTH.md`,
*Modulation*): the measure is the source, 0..1, and the slot's base and amount carry the rest,
the range and the direction. A measure may feed several parameters; a slot has one source; two
measures never sum into one parameter.

The arms are `MATRIX.md`'s Option 3. The bases and amounts are the preset's starting values,
tuned on the machine:

| Source              | Range | Parameter         | Base    | Amount     | At full                                        |
|---------------------|-------|-------------------|---------|------------|------------------------------------------------|
| white width         | 0..1  | white pulse width | 0       | 1          | solid white                                    |
| blue width          | 0..1  | blue pulse width  | 1       | −1         | no blue                                        |
| left elbow          | 0..1  | white pitch       | 25.7    | 46.3 lines | 72 lines, one every 5°: finer as the arm folds |
| right elbow         | 0..1  | blue pitch        | 25.7    | 46.3 lines | 72 lines                                       |
| left turn + bend    | −2..2 | white speed       | 3.9°/s  | 15°/s      | white flowing one way at +90°, back at −90°    |
| right turn + bend   | −2..2 | blue speed        | −4.5°/s | 15°/s      | as white's                                     |
| leg deviation       | 0..1  | the LFO's level   | 0       | 1          | the LFO at full swing; the LFO feeds nothing   |

The widths are the shoulders' mean with the breath on the higher shoulder's colour:

```
white width = shoulders + depth × left excess  × breath       left excess  = max(0, left − right)
blue width  = shoulders + depth × right excess × breath       right excess = max(0, right − left)
```

The breath is a sine in time, −1..1, one per person, at `PI.breath.rate`; the depth is
`PI.breath.depth`. Level shoulders have no excess, so the fixed points and a T are still; a depth
of ½ or less keeps every breath inside none and full, so both fixed points stay exact
(`MATRIX.md`, Option 3). An elbow's turn is the sine of its signed angle: still when straight and
when fully folded, where +180° and −180° are one pose. The body bend, signed, is
added to both turns with one sign: white drifts outward and blue inward, so a lean one way makes
the white faster and the blue slower, the other way the reverse. A turn and a lean share the
amount, so a forearm and a lean can add or cancel. Unconnected: the LFO, the symmetries, the
distance, the phases, the hardness.

## Events

What the instrument triggers in a person's voice, beside the measures:

| Event    | When                                                        | Acts on                                                     |
|----------|-------------------------------------------------------------|-------------------------------------------------------------|
| presence | the person is seen: a gate open while they are there        | the presence envelope, on both reaches                      |
| hit      | the playhead crosses the person: the tick closest to it     | each oscillator's push, and the mask's flash                |
| sync     | the similarity of a pair is over its threshold              | the reach on the partner's side: full reaches the partner   |

Presence and the pushes are envelopes of the synth (`LIGHT_SYNTH.md`, *The envelope*). The
reaches are the bridge's: it gives each side's reach to the voice as a value, the rest width grown
toward the partner by sync, since only the bridge knows where the partner stands. The mask's
flash is the bridge's own envelope of the same block, opened by the hit at once and falling back
over its release: the mask goes to its flash levels and returns to its own. The mask is the
bridge's and not the synth's.

## Consequences

What follows from the connections, before the machine has been judged **(deductions)**:

- Level shoulders tile the wall: white's width and blue's add up to the interval and blue sits half
  an interval from white, so every pixel is one colour, from full blue through the T to full white.
  With one shoulder higher its colour breathes against the still other, so the tiling opens and
  closes: the overlap tone as it swells, dark as it thins; which colour breathes says which arm.
- The elbows' symmetry shows by itself, unconnected: equally folded, white and blue share one
  interval and stay in tune; unequally, the colours slide past each other with distance
  (`LIGHT_SYNTH.md`, *In the pose instrument*).
- An elbow shows nothing while its own colour is solid or dark: the pitch needs a note.
- An elbow's turn, like its fold, shows nothing while its colour is solid or dark.
- Unequal elbows also open the tiling, as a moiré that changes along the wall and travels; the
  breath opens it in time, evenly along the wall.

Once the connections have been played on the machine, the *Pose results* are read back here: what
each row draws, against what it should.

---

# Part 4 — Implementation

## Design

The bridge gives every person a voice of the light synth, sends its two outputs to white and to
blue, feeds the measures of *The body* into the slots of *The connections*, triggers the *Events*,
and draws the mask over the result. The synth, its building blocks and its rules are
`LIGHT_SYNTH.md`'s.

## The code

### The layer

`pose_instrument` (`light/layers/projection/pose_instrument.py`) is the bridge. It gives each
person a `Voice` of the light synth (`light/synth`), paints the voice's output 1 into white and its
output 2 into blue over the person's window, and draws every mask over the result. Overlapping
voices combine per channel, the fuller showing; two patterns of different intervals or centres
make a moiré. Distances are taken from the person's own azimuth and not from their centre pixel,
so a walking person's lines move smoothly. With `opposite` the pattern is drawn half a turn from
its person while the mask stays on them; the masks still go over every pattern, so they cut the
patterns that fall on them. Sync still grows a reach toward the partner as seen from the person,
so with the switch on a pair's patterns open away from each other. The instrument draws the playhead's marker over the
masks (`PlayheadMarker`, `playhead_marker.py`), and the render shows the overlap of the two
colours as a tone of its own (`render/shaders/lightsimulation.frag`). A tick costs about 0.2 ms
per person at 3600 pixels. Its place in the states' mixes is in `LAYERS.md`, *pose_instrument*.

### Sources and connections

The layer reads its sources from the LERP frames: the elements of `ArmTravel` (the four arm
joints, the absolute taken), the elbows' elements of `Angles` (the turn), `LegDeviation`,
`TorsoTilt`, `Distance` and `AngleSymmetry` (*The body*, *Symmetry*).

The connections are code, not settings: `PoseInstrument.connect` takes a person's measures, the
dead zones already on them (*The body*), and returns the sources of the white and the blue
oscillator's slots. It is *The connections* written out. `connect_lfo` beside it gives the LFO's
level its source and runs first each tick, so `connect` can read the LFO's output. The bases and
the amounts, the range and the direction of every connection, are settings: the panel keeps the
values and the code keeps the routing.

### People

A person is present while their pose exists (`LAYERS.md`, *Inputs*); a pose with a NaN azimuth
has no place in the projection and counts as absent, the one presence test a layer adds.
Presence opens the window from the mask over `window.attack_seconds` and closes it over
`window.release_seconds`, the last pose held while it closes and the mask dimming with it.

The bridge sets each side's **reach** every tick: `window.width`, grown by sync toward every
partner along the shorter arc. Sync starts at `window.sync_threshold`, on the mean of both
directions' similarity; the threshold is the bridge's remap of the similarity, which is well above
0 for most pairs out of neutral (`SIMILARITY.md`, *Interdependence*). The reach grows eased until
it meets the partner at similarity 1: full sync is full overlap, one pattern. Only the partner's side opens, over any
person between them, whose own pattern is unchanged. The partner's presence scales the growth, so
a partner leaving lets go smoothly. A pair with a person at neutral reads 0 and does not open
(`STATES.md`, *Vocabulary*). The voice multiplies the reach by presence and thins the lines to
nothing over the taper (`LIGHT_SYNTH.md`, *The window*).

The **mask** is a band `mask.width` wide at the person, each channel at its level (`mask.white`,
`mask.blue`) times presence, and goes over everything at the person: the patterns of every voice.
A mask cuts a line where it falls, so lines slide out from under it; the window cuts nothing. In
the preset the white is 0, so the mask is the dim blue band. The playhead's **marker**
(`PI.playhead`: its width, its white and blue) is drawn by the instrument over the masks, dimmed
to `playhead.at_mask` inside one, so the marker never blinds and no other layer shares a mix with
the instrument at a person.

### The hit

On the tick the playhead is closest to a person (`PlayheadCrossing` in `pose/playhead_offset.py`,
`PoseInstrument.HIT_TICKS`, the same rule as `beam_flash`), each of the person's oscillators is
pushed (its `push`, falling back over its `push_release_seconds`) and the mask goes to its flash
levels (`mask.flash_white`, `mask.flash_blue`), falling back to its own over
`mask.flash_release_seconds`. Every mark has a white and a blue level, and 0 is off. The crossing
is measured in playhead steps at `beam_rpm`, the rate the playhead free-runs at in PROJECTION.

### Settings

`PI` is a root settings group of the app (`apps/white_space/settings.py`), not a layer group
(`PoseInstrumentSettings`, `light/layers/projection/pose_instrument.py`), grouped by what is
tuned together, knobs throughout:

| Group                       | What it holds                                                   |
|-----------------------------|-----------------------------------------------------------------|
| `max_lines`                 | the visual limit: the pitch ceiling                             |
| `opposite`                  | draw the patterns half a turn from their people                 |
| `mask`                      | the mask's width, white and blue; the flash's levels, release   |
| `playhead`                  | the marker's width, white and blue; its level inside a mask     |
| `window`                    | shape: taper, attack, release; reach: width, bypass, sync       |
| `breath`                    | the breath's rate and depth (*The connections*)                 |
| `white_lines`, `blue_lines` | On, Mirror, Bypass All; a slot per parameter; the push          |
| `lfo`                       | the LFO: rate, phase, the level's slot                          |
| `dummy`                     | *The dummy*                                                     |

The groups run from the person outward: the mask at the person, the marker that passes them, the
window around them, then what fills the window, then the tool. The dead zones on what the body
gives are the pipeline's settings (*The body*). Wherever a
mark has a level per channel the two settings are `white` and `blue`.

Every row of the panel is titled. A slot's row is titled with its parameter's name and reads
Base · Amount · Curve · Bypass (`LIGHT_SYNTH.md`, *Modulation*).

The synth's settings classes (`OscillatorSettings`, `LfoSettings`, `WindowSettings`) are extended
by the bridge's where a concept spans both (`window`); the voice reads only its own fields. The
layer keeps its `blend`. No setting routes anything: an amount does nothing until `connect` gives
its parameter a source.

### Hot reload

The composition is worked on the machine in two ways. Values are tweaked in the panel and saved in
the preset. Connections are edited in code and seen at once: `HotReloadMethods` (`modules/utils`)
watches `pose_instrument.py` and the synth's files and, on save, re-executes their class bodies
and module constants into the running app, so `connect`, the drawing methods and the synth's
classes change under the layer without a restart.

What the reloader patches: methods of a class (instance, static, class), class constants, module
constants. What it does not: module-level functions, imports, settings fields, and the objects
already built (a settings instance keeps its fields). Hence the rules:

- Everything that may change lives in a class method or a module constant, never in a module-level
  function.
- The synth's `Parameter` is an `IntEnum`, so a source's key still matches after a reload redefines
  the class.
- The numbers of a connection are its slot's base and amount, settings, never literals in
  `connect`, so the preset keeps describing the show.
- A new setting, feature or parameter needs a restart; a new connection does not.
- The table of *The connections* is `connect` in words; when one changes, the other follows in
  the same change.

### Playing by hand

A parameter's base is already the hand's value, so playing by hand is bypassing modulation. Every
slot has a **Bypass** (`PI.white_lines`, `PI.blue_lines`, and the level of `PI.lfo`): bypassed,
the parameter is its base while every other parameter keeps following the body, so a pose can be taken apart
parameter by parameter; lifted, it follows again with its amount as it was. An oscillator's
**Bypass All** button sets its five ticks at once, and pressed again with all set clears them, so
the panel draws that colour for everyone, live people and the dummy alike, while the other colour
and the LFO keep following the body; the ticks are the one state, and can be changed one by one
after. An oscillator's **On** switch, off, draws nothing for that colour.
`window.width_bypass` holds both reaches at the width without a partner, presence still opening
and closing them. What the body gives is read in the render's data graphs, per person.

### Tests

`apps/white_space/tests`: the synth in `test_synth_oscillator.py` (among them the no-jumps rule,
each parameter swept in small steps), `test_synth_envelope.py` and `test_synth_voice.py`; the
bridge in `test_pose_instrument.py`; the dummy in `test_dummy.py`.

### The pose results now

The rows of *Pose results* draw what that table says, in the unit tests
(`tests/test_pose_instrument.py`: the two fixed points, exact at any breath, one shoulder moving
both colours, a T still over a breath, the higher shoulder breathing its own colour, left and
right up drawn differently, level shoulders tiling, an elbow making its own colour finer, equal
elbows in tune, an elbow's turn flowing its own colour either way, a lean making the white faster
and the blue slower and back, and a small move of any arm measure a small change of the picture). What they draw on the machine has not been judged yet.

### The dummy

The dummy (`pose/dummy.py`, settings `PI.dummy`) stands in for a person while the instrument is
judged: a figure whose joints and torso lean are set in the panel, each joint in degrees
−180..180 from its rest (arms hanging, elbows and knees straight, standing). Its frame joins
the poses before the LERP filters at its own id, `max_players`, so from there it is a pose like
any other: extracted, drawn, heard in Max and lit. With `solo` it is the only pose from the LERP
filters on. The preset's calibration is the dummy's raw readings,
so its hanging arm reads 0 and its vertical arm π. How the figure is built is the module's
docstring.

Its poses are named in `data/poses.json`, the rows of *Pose results*, picked in the `pose`
select and saved under `name` with `save`. A change of any measure morphs over `morph` seconds.

---

## Open

The open questions of the synth itself are in `LIGHT_SYNTH.md`, *Open*.

Meaning and connections:

- Whether the connections of Part 3 can be found with the body: the shoulders' balance and
  breath, the elbows' pitch and flow, the lean; what the legs and the symmetries should play
- Which way is "right": a positive `TorsoTilt` is shoulders toward image right; whether that
  lean makes the white faster on the wall as meant is to be checked on the dummy
- The breath's rate and depth, and whether a breath reads as the higher arm's
- Which of the instrument's departures from the rules of meaning stay, and which calculations move
  into the pose pipeline, once the result is liked
- The lean moves the hips' reading by the lean, on the dummy as on a person, so at full lean the
  leg deviation reads about 0.75 and whatever it plays sounds with the bend; whether the leg
  deviation should be taken against the vertical instead of the torso line
- The symmetries: which of the four pairs earns a connection, and what it means
- The distance: what it plays, and whether the reading is steady enough to play it; it comes off
  the detection box's bottom row and is weakest at the zone's far edge (`TRACKING.md`, *The pose's
  distance*)
- The composition itself: which connections people find with their bodies
- Whether the wiring of a connection is chosen in the panel or stays code, as `connect` is

Not yet in the design:

- Tuning between people: intervals in simple ratios, so that synced people read as a chord
- Oscillator sync between people: one pattern's grid adopting the other's where they meet, an
  alternative to the union
- Expression: the motion features (angle motion, motion time) as sources, a fast arm giving a
  stronger push
- Decay and sustain on the window: the arrival flourish

From the machine **(site facts)**:

- Whether the preset's mask width and window width are right, and what the *Pose results* draw;
  judged on location
- Whether the 90-line limit holds with the moiré of two synced patterns
- A pitched wide-FOV camera reads a small spurious body bend near the frame edges, a bend nobody
  played; check a straight person at the edge
