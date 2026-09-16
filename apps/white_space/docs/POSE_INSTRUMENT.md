# White Space — Pose Instrument

The instrument the participants play with their bodies: the light half of pose → sound. Four
parts. **Vocabulary** is the language of the instrument, borrowed from the synthesizer. **Meaning**
is what each measure of the body should say in it. **Connections** is how the measures are wired
to the pattern so that the meaning comes out. **Implementation** is the layer's design and its
state today.
Meaning and connections are one loop: meaning sets the targets, the targets pick the connections,
the connections have consequences, and the consequences are new meaning to judge. While one is
held still the other is worked on.

The layer's place among the other layers is in `LAYERS.md`; the states that run it are in
`STATES.md`. The order the design is built in, each step visible on the machine: the sources and
the connections; the pattern; motion and marks; the playhead at the masks and the render.

## The instrument

Each person is a source of light in the projection. At their azimuth sits a dim blue **mask**: it
goes over everything at the person, the patterns of everyone and the light of the other layers;
the playhead passes through it dimmed by a setting. Around the mask, mirrored about the person,
their **pattern**: lines of full white and full blue, generated behind the mask and coming out from
under it. Only a **window** of the pattern is visible each side of the person; sync opens the
window wider.

The ground rules, as set:

- About 90 lines per revolution, line and gap equal, is the visual maximum **(site fact)**. It is a
  guide: a good reason may go past it.
- No brightness: every pixel of white and of blue is off or full. The mask is the one dim
  thing.
- No blinding: the instrument never blinds a person with white; people must be able to see each
  other. The mask is how, so full white can fill a window and never fall on a face. This rule is
  the instrument's alone, not the other layers'.
- White and blue are projected separately by the fixture **(site fact)**, so where both are on the
  overlap reads as a subtly different tone. The palette is four tones: dark, blue, white, both.
- No jitter: the pose pipeline's smoothing is the only smoothing. The layer adds no smoothing and
  no steps, since steps are jitter of another kind, and no flicker of its own.
- Lines follow the pose. On top of that they drift by themselves, slowly and peacefully, blue
  inward and white outward: a low setting that may be 0 (see *Drift*).
- The output each side of a person is symmetric.
- The pattern is LFO based; the six pose values control it; every combination of the six draws
  differently; the arms come first, legs and body bend second.
- Arms down is neutral: full blue over the window. Arms up: full white. In between, the pattern.
- Blue and white can behave differently; line thickness and interval can change.
- A hit (the playhead crossing the person) marks that person for one frame. Wider lines do not
  read **(site fact)**; the mark is *The hit*.
- Sync makes more of the pattern visible: the window opens, the lines stay what they are.

---

# Part 1 — Vocabulary

## The oscillators

The pattern is two low-frequency oscillators, one for white and one for blue. They are spatial:
they oscillate over the distance from the person, not over time. The vocabulary is the LFO's.

The projection is an oscilloscope of one dimension. It draws sine waves, but it has no height to
draw them with, so it draws where the wave rises above a level: a bar for every crest. The
thickness of a bar is how long the wave stays above the level; the interval is the wavelength.

| Synth term     | In the instrument                                                         |
|----------------|---------------------------------------------------------------------------|
| waveform       | sine, triangle or saw, thresholded into bars (see below)                  |
| interval       | the distance between lines, in degrees: the LFO's rate, spatial           |
| detune         | blue's interval leaving white's: 0 locked to one grid, more and they beat |
| pulse width    | the thickness of a line as a fraction of the interval: 0 none, 1 solid    |
| phase          | where the lines sit relative to the mask                                  |
| cutoff         | which overtones pass: none, or up to 2, 3 or 4 per interval               |
| resonance      | how strongly the passing overtone is emphasised: sub-lines, uneven lines  |
| overtone phase | where the overtone sits within the interval                               |
| drift          | own motion: the phase advancing over time, blue inward, white outward     |

A **filter** shapes the wave before the level: the cutoff says which overtones pass, the resonance
how strongly the passing overtone speaks. Open, the bars split into sub-lines or grow uneven;
closed, there is a plain bar per crest. The filter is additive: the overtone is a second wave at
the cutoff, mixed in before the level. An overtone is not a second oscillator: it sits at a whole
number of cycles per interval and can never leave its fundamental, so the pattern still repeats
every interval. Only detune lets one wave leave another. The **waveform** decides
how thickness answers the level: a sine or a triangle grows a bar from its centre both ways, the
triangle linearly; a saw grows it from one edge only, toward the person or away from them.

**Registration** is the organ's word for which drawbars are out. The fundamental's drawbar is the
pulse width, the harmonic's drawbar is the resonance: one thickens the lines, the other the
sub-lines. Both drawbars out fills the window with that colour; both in is silence in it.

Each colour has its own oscillator: its thickness, phase, filter and drift are its own. The
interval is one, shared, and **detune** lets blue's interval leave white's; at 0 the two are locked
to one grid. Locked, with phases half an interval apart, they alternate white and blue. With equal
phases they stack into the overlap tone with dark between. Detuned, the two beat: a moiré within
one person's pattern. The colours meet as AND, the overlap tone where both are lit; they could also
meet as XOR, dark where they coincide, lines that cancel: ring modulation.

## Drift

**Drift** is the lines' own motion: the phase advancing per second, so the oscillator has two
rates, the interval over distance and the drift over time. Phase is where the lines sit; drift is
how fast they travel. It is a low setting, possibly a connection (the body bend, or something else,
could set its speed): blue toward the person, white away from them.

A **push** is a drift event: for a moment the drift speeds up, the lines travel faster, then it
settles back to the slow drift. The lines keep the distance they gained; nothing comes back.

## The voice

Each person is a **voice**; the number of people is the polyphony. Presence is the note: attack
when a person arrives and release when they leave, both on the window. A **decay** and a
**sustain** would add an arrival flourish, the window opening wide and settling to its rest width.
The push is an attack-release envelope on the drift, and the accent may be a trigger with a short
decay if one frame proves too short to see.

**Expression** is how fast and how hard a key is played: velocity and aftertouch. The pose pipeline
measures motion (angle motion, motion time), and those are sources like any feature: a fast arm
can give a stronger push or a wider tint.

**Unison** is sync: two voices stacked. Two nearly equal intervals in unison beat, which is the
chorus of unison detune: the moiré, and the reason it must stay small.

## The hit

The playhead crosses a person once per revolution of the content sweep. On that frame the person is
marked: the accent. The mark lasts the closest frame, or up to the three closest, a setting as the
beam flash has, and has to read within the palette of four tones. Candidates, each per colour where it applies:

| Mark         | For the frame                                                        | Settings                          |
|--------------|----------------------------------------------------------------------|-----------------------------------|
| tint         | each colour's lines take the other colour: blend 0 none, 1 swap     | blend 0..1 per colour             |
| mask flash   | the mask goes brighter blue; white is beam mode's flash              | the blue level                    |
| push         | the drift speeds up for a moment and settles back                    | strength, settle time             |
| window pulse | the window opens wider for the frame: more pattern shows             | the extra width                   |

Tint keeps the pattern and changes its colour, and at blend 1 it is the swap; the mask flash is
the person's own light; push and window pulse speak the drift and sync vocabulary. Chosen to try:
tint, push and the mask flash. The layer's mark today is in Part 4.

---

# Part 2 — Meaning

## What we hold on to

Two meanings are fixed, the ground rules' neutral and raised arms: the blue ping and the bass, as
the sound has them (`STATES.md`, S4: a glass ping for neutral, a heavy bass for arms raised).
Between the two the magic happens, and this part describes it: a pattern for every combination of
the measures, so each measure of the body has to mean something, and nothing but the two fixed
points is tied to a pose.

## The rules of meaning

1. Only measures mean something: the six measured values, and what the pose pipeline derives from
   them (the symmetry of a pair). The instrument derives nothing.
2. A meaning belongs to a measure, never to a pose. The table says what happens when a measure
   changes. No pose owns a meaning; the two fixed points are the only exceptions.
3. Groups of measures (both arms, both elbows) and poses are consequences of the measures'
   meanings. They are read back in Part 3, never written here.
4. Raising one arm never means the same as raising the other, and the two arms are connected by a
   relation from the vocabulary of Part 1: not by a colour, and not by whatever two parameters were
   free.
5. The same holds for every pair of joints: a pair acts through a musical relation, or through its
   symmetry.
6. A meaning is spoken in the vocabulary: it names a synth term.
7. Every combination of the measures draws differently.
8. Few connections, much meaning. A measure may mean nothing on its own; then its symmetry may.
   Not every parameter of the vocabulary needs a measure.
9. Primary measures make the note, secondary measures modulate it. A secondary meaning needs a
   note to act on, so it is judged on a pattern, never on the fixed points.

## What each measure could mean

Each measure is a musical term. The first best guess, with the alternatives beside it.

**Primary** measures make the note: the arms are the organist's hands on the registration, the
left the fundamental's drawbar and the right the harmonic's, pulling the white out and the blue in;
the relation between them is the harmonic series, a note and its overtone; each elbow places what
its shoulder pulls. **Secondary**
measures modulate the note: detune, pitch bend and drift are what a player does to a note that is
already sounding, and with the arms hanging there is no note to modulate. The drift is a setting
until a secondary measure takes it.

| Measure           | Term                      | Role      | Or                                  |
|-------------------|---------------------------|-----------|-------------------------------------|
| left shoulder     | the fundamental's drawbar | primary   |                                     |
| right shoulder    | the harmonic's drawbar    | primary   |                                     |
| left elbow        | the fundamental's phase   | primary   | the interval                        |
| right elbow       | the overtone's phase      | primary   | the overtone's order                |
| leg deviation     | detune                    | secondary | the interval; the drift             |
| body bend         | pitch bend                | secondary | the drift's speed; the phases apart |
| shoulder symmetry | unconnected at first      | secondary | a solo; a phase apart               |
| elbow symmetry    | unconnected at first      | secondary | syncopation                         |

Two events sit over every measure: the **accent** of a hit (*The hit*) and the **unison** of sync,
the windows of two people opening toward each other until the two patterns overlap and become one.

## Composition

The aim is a balanced composition that people recognise they influence. Not every parameter of the
vocabulary has to be used; they are there to be tried, because several may mean the same thing to
the eye, or the pattern's own workings may already produce what a parameter would add. A connection
earns its place when a person can find it with their body. Few connections, much meaning: a feature
may drive more than one parameter, and a parameter may stay fixed.

---

# Part 3 — Connections

## Pose results

The poses we check against: each with the light it should produce under the meanings of Part 2.
The list is the acceptance set, small on purpose; on the machine each row is a thing to stand in
and look at, and a test. The rows are the dummy's saved poses (`data/poses.json`, *The dummy*),
so each is one pick in the panel. The first two are the calibrator's reference poses, `neutral`
(arms hanging, standing) and `raised` (arms up); the others are built on them, the T halfway along
the calibrator's shoulder arc from neutral to raised, so it reads half registration by
construction. Every pose between the rows is unique and is not described.

| Pose                                        | Result                                                         |
|---------------------------------------------|----------------------------------------------------------------|
| neutral                                     | full blue over the window: the blue ping                       |
| raised                                      | full white over the window, no blue: the bass                  |
| arms out level, a T                         | white lines with sub-lines, blue between: half registration    |
| left arm up, right hanging                  | thick white lines, no sub-lines: the fundamental alone         |
| right arm up, left hanging                  | thin white sub-lines only: the harmonic alone                  |
| a T, both elbows folded                     | the lines and the sub-lines moved out from the mask            |
| a T, left elbow folded                      | the lines moved out, the sub-lines in place                    |
| a T, right elbow folded                     | the sub-lines moved off the lines                              |
| a T, leaning                                | the whole pattern's interval bent, back when upright           |
| a T, in a crouch                            | blue lines beating against the white: detuned                  |

The elbow and the secondary rows stand on the T because the elbows place white lines and the
secondaries modulate them: with the arms hanging there are no white lines to place or modulate, so
hands on the hips draw the blue ping and hands on the head draw the bass.

## The body

The instrument's sources are pose features and nothing else: values the pose pipeline measures
and publishes with every frame, smoothed. The instrument never derives a value of its own. The one
derivation the pipeline adds is the symmetry of a pair, as it derives the leg deviation from the
hip and knee angles.

| Feature           | Measure                                                  | In the pipeline |
|-------------------|----------------------------------------------------------|-----------------|
| left shoulder     | 0 hanging → ±π straight up, the sign the side it passes  | `Angles`        |
| right shoulder    | 0 hanging → ±π straight up, the sign the side it passes  | `Angles`        |
| left elbow        | 0 straight → ±π folded                                   | `Angles`        |
| right elbow       | 0 straight → ±π folded                                   | `Angles`        |
| leg deviation     | 0 standing → 1 bent, stretched                           | `LegDeviation`  |
| body bend         | −1 left → 0 upright → 1 right                            | `TorsoTilt`     |
| symmetry, a pair  | signed, left minus right: how unequal the two sides are  | `AngleSymmetry` |

The angle extractor measures the geometric angle between body segments, the arm against the
torso line, mirrored on the right so a symmetric pose reads equal on both sides; what a body
reads at neutral is its geometry, and a vertical arm is not π from it. The angle calibrator
(`modules/pose/nodes/filters/AngleCalibrator.py`, settings `pose.angle_calibrator`) then maps
each joint from two reference poses, the body with the arms hanging and with the arms raised,
each held as the raw readings: every reading at neutral becomes 0; the shoulder's raised reading
becomes π, so the fixed points are the feature's 0 and π for sound and light alike; the elbow's
raised reading is where its straight is with the arm up, so a relaxed straight elbow reads 0 in
both poses and π is the fold from there. The readings are tuned in the panel, against a person or
the dummy. The sign of an angle is the side of the body the limb
passes, outward or across, and means nothing to the instrument: the connections take the
absolute, so an arm raised outward and one raised across the body read alike.

A **connection** is one feature into one pattern parameter. A feature may feed several parameters;
a parameter has one source; two features never sum into one parameter. The connections are a
**modulation matrix**: features down one side, parameters along the other, a depth and a polarity
in each cell; a feature feeding several parameters is a macro. The matrix is code (Part 4,
*Sources and connections*); the panel holds the values it uses.

## Symmetry

The light is mirrored, so left and right in the body never show as left and right in the light;
which arm is which shows only as a different sound. The **symmetry** feature (`AngleSymmetry`)
holds the meaningful pairs, the four joints (shoulder, elbow, hip, knee) and the arm and the leg as
a whole per side (`arms`, `legs`), each signed, left minus right, so that downstream the signed
value or its absolute can be used. A person and their mirror image draw differently.

## The first connections

The connections that make the meanings of Part 2 come out: feature into pattern parameter. The
two drawbars are the same for both colours, inverted: what pulls the white out pushes the blue in.
The bass is a consequence: one drawbar lights at most half of each interval, both at full light it
all (the registration, Part 4 *Design*).

| Feature           | Connection                                               | Alternatives                                                   |
|-------------------|----------------------------------------------------------|----------------------------------------------------------------|
| left shoulder     | white fundamental 0 → 1, blue fundamental 1 → 0          |                                                                |
| right shoulder    | white harmonic 0 → 1, blue harmonic 1 → 0                |                                                                |
| left elbow        | white phase 0 → ½, outward                               | the interval                                                   |
| right elbow       | white overtone phase 0 → ½                               | white cutoff 2 → 4                                             |
| leg deviation     | the detune 0 → its maximum                               | the interval; the drift                                        |
| body bend         | the interval, scaled by ± its octave range               | the two phases in opposite directions; the drift               |
| shoulder symmetry | unconnected                                              | one colour's interval or phase                                 |
| elbow symmetry    | unconnected                                              | the two phases against each other                              |

## Consequences

Once the first connections run on the machine, the *Pose results* are read back here: what each
row draws, against what it should. Empty until then; what the current build draws is in Part 4,
*The pose results now*.

---

# Part 4 — Implementation

## Design

The layer built to Parts 1 to 3. Where a choice is still open it is marked **(open)**.

### Sources and connections

The instrument reads pose features from the LERP frames and derives nothing. The sources are the
elements of `Angles` (the four arm joints), `LegDeviation`, `TorsoTilt`, and `AngleSymmetry`
(`modules/pose`): one signed scalar per pair, left minus right, normalised to −1..1, made by its
own extractor (`AngleSymExtractor`) and windowed, graphed and sent to Max like the leg deviation.
The pairs: the shoulder, the elbow, the hip and the knee (the joint angles), the arms (shoulder and
elbow together per side), the legs (hip and knee together per side, weighted as the leg deviation
is, by its settings).

The connections are code, not settings: one method, `PoseInstrument.connect`, takes a person's
measures and returns the pattern parameters of both colours (`Pattern`: the interval and the
detune, and an `Oscillator` per colour: the two drawbars and the two phases). It is the first
connections table of Part 3 written out, with the ranges, the curves and the absolutes of signed
values that a panel cannot express, and it hot-reloads on save. Every number it uses is a setting
it reads (the rest interval, the detune's maximum, a phase's range), never a literal, so the
panel keeps the values and the code keeps the routing. The preset carries values only. An angle
becomes a measure as its absolute over π, the feature's range, since the calibrated angle is 0 at
neutral and π at raised, and the sign is the side the limb passes and not a measure.

### The pattern

Per person, one **interval** in degrees, shared by both colours: a rest value from settings, bent
by its source (the body bend, see the pitch bend below). **Detune** makes blue's interval
`white × (1 + detune)`; at 0 the colours share one grid.

Per colour: `waveform` (sine, triangle, saw), the two drawbars `fundamental` and `harmonic` (each
0..1), `phase`, `cutoff` (2, 3 or 4), `overtone_phase`, `drift`. The colour's wave is the filter
of Part 1, the two drawbars as its weights, thresholded at one level:

```
u    = x / interval − phase                          phase positive moves the lines outward
wave = (fundamental · w(u) + harmonic · w(cutoff · u + overtone_phase)) / (fundamental + harmonic)
lit  = wave ≥ 1 − (fundamental + harmonic)
```

`w` is the waveform, a cosine for the sine. The wave is the mix of the two, so it stays within
−1..1 whatever the drawbars; the level falls from 1 with both drawbars in to −1 with both out. One
drawbar alone lights at most half the interval (the fundamental thick lines, the harmonic thin
sub-lines); both out light everything. The bass is solid; one drawbar is not.

The blue is the same instrument with the registration inverted: the drawbar that pulls the white
out pushes the blue in, so at rest the blue is full and at the bass it is silent. The blue's rest
phase is half an interval from the white's, so the colours interleave when locked.

The **pitch bend** scales the interval: `rest × 2^(bend × octaves)`, with `octaves` the range each
way, so a lean to one side halves the interval at most and to the other doubles it.

**Drift** advances each colour's phase by `drift` intervals per second, blue negative (inward),
white positive (outward); the phase's source adds to it. A **push** raises each colour's drift by
`push_strength` in that colour's direction on the hit and lets it settle back over
`push_seconds`; the phase keeps what it gained.

The visual limit is `max_lines` (90): no line and no gap narrower than half a period of it, 2° at
90. The interval never goes below one period; a line or a gap narrower than the limit is filled or
dropped as the current layer does. A low pitch thins the harmonic first: below `cutoff` periods
its sub-lines fall under the limit and drop, as a low note loses its overtones. A mask or the
window's edge cuts a line as it is, so lines slide out from behind the mask and into view at the
window.

### People

Each person's pattern is mirrored about their centre pixel and cut to the **window** each side:
`window` degrees × the presence envelope (attack from 0, release to 0), grown by sync toward every
similarity-matched partner along the shorter arc, up to the partner. Patterns union per colour and a
gap narrower than the limit fills. On arrival the mask is there at once and the window opens; on
release the window closes first and then the mask fades: the pattern goes first, the person's own
light last.

The **mask** is a dim blue band, `mask_width` wide at `mask_brightness`, and
goes over everything at the person: the instrument's own patterns, and the playhead. The projection
playhead dims itself to `playhead_at_mask` inside a mask: it reads the poses and the mask width as
the instrument does, so there is no store and no compositor change. No other layer shares a mix
with the instrument at a person.

### Events

On the frames the playhead is closest to a person (`PlayheadCrossing`, `hit_frames` 1 to 3, as the
beam flash), the person's mark: a
`tint` blend per colour, 0 none, 1 the swap; between 0 and 1 the central fraction of each line
takes the other colour, never narrower than the limit, and a line whose rims would be narrower
than the limit is taken whole. The `mask_flash` raises the mask's blue to `mask_flash_brightness`
for the frame. The push is above.

### Settings

`PI` is a root settings group of the app (`apps/white_space/settings.py`), not a layer group, and
holds tweakable values only, each concern a `Group` (`PoseInstrumentSettings`,
`light/layers/projection/pose_instrument.py`): `max_lines`; `pattern`, the rests and ranges (the
interval at rest and its octaves of bend, the detune's maximum, the blue's rest phase, the phases'
ranges, and per colour `white` / `blue` the waveform, the cutoff and
the drift); `mask` (width, brightness, the playhead's level in it, the flash brightness); `window`
(width, the sync threshold); `events` (the hit's frames, the tint per colour, the push); `presence`
(attack, release); `override` (*Playing by hand*: the pattern's parameters set directly, a toggle
each and one `on`, the window held, a hit button); `dummy` (*The dummy*). The layer keeps its
`blend`. No setting routes anything.

### Hot reload

The composition is worked on the machine in two ways. Values are tweaked in the panel and saved in
the preset. Connections are edited in code and seen at once: `HotReloadMethods` (`modules/utils`)
watches `pose_instrument.py` and `line_pattern.py` and, on save, re-executes their class bodies and
module constants into the running app, so `connect`, the drawing methods and `LinePattern` change
under the layer without a restart. The panel is for tweaking; making connections needs ranges,
curves and the absolutes of signed values, which a panel of fields cannot express and a node
editor could, and that is out of scope.

What the reloader patches: methods of a class (instance, static, class), class constants, module
constants. What it does not: module-level functions, imports, settings fields, and the objects
already built (a settings instance keeps its fields). Hence the rules:

- Everything that may change lives in a class method or a module constant, never in a module-level
  function.
- A reload redefines the enum classes, so enum values are compared through `int`, never `==`.
- Every number `connect` uses is a setting it reads, never a literal, so the preset keeps
  describing the show.
- A new setting, feature or parameter needs a restart; a new connection does not.
- The first connections table of Part 3 is `connect` in words; when one changes, the other follows
  in the same change.

### Playing by hand

The instrument has two kinds of numbers. Its **inputs** are what the pipeline delivers per person:
the arm angles, the leg deviation, the bend, the similarity, the playhead crossing; the dummy fakes
them. Its **parameters** are what the pattern is drawn from: the interval and the detune, and per
colour the two drawbars, the phase and the overtone phase, the `Pattern` that `connect` returns
for a person each tick. `PI.override` sets the parameters directly: with `on`, every ticked
parameter comes from the panel for everyone, live people and the dummy alike, and its connection
is skipped; an unticked one keeps following its measure, so a pose result can be taken apart
parameter by parameter. `on` sets nothing else; the toggles are independent. `window` holds the
window each side without a partner, presence still opening and closing it; `hit` marks everyone
for `hit_frames` ticks as the playhead would. Waveform, cutoff, drift, the mask and the hit's
values are settings already.

### Tests

The pose results of Part 3 are unit tests in `apps/white_space/tests`: a pose's features into
`connect`, the parameters into the pattern, one test per row.

## State

### The layer

`pose_instrument` (`light/layers/projection/pose_instrument.py`) draws one boolean mask per colour,
mirrored about each person's centre pixel, cut to the window each side, unioned over people, masked
by every mask, and written as 0 or 1 into the frame. How people compose (union, visibility,
the masks, sync growth, the hit, presence) is in `LAYERS.md`, *pose_instrument*. The sources, the
connections (`connect`) and the `PI` settings group are as *Design* has them; the sources are the
features of *The body*, and the symmetries are read and connected to nothing.

The line math is `LinePattern` (`line_pattern.py`), *The pattern* as written: the wave is the mix
of the fundamental and its overtone at the cutoff, weighted by the two drawbars, thresholded at
the level; the waveform is the colour's setting (sine, triangle, saw). The interval is one per
person, shared by both colours, bent by the body bend and detuned for the blue.

The visual limit is `max_lines` (90): no line and no gap narrower than half its period. The
interval never goes below one period; a line or a gap narrower than the limit is filled or dropped
(`LinePattern.visible`). A mask or the window's edge cuts a line as it is, so lines slide out from
behind the mask and into view at the window.

Drift and the push are as *The pattern* has them, the push settling exponentially over
`push_seconds`; the phase keeps what it gained. The hit (`events.hit_frames` frames) marks the
person as *Events* has it: the tint per colour (`LinePattern.core`), the mask flash, the push. The
projection playhead dims itself at the masks as *People* has it (`projection_playhead.py`), and the
render shows the overlap of the two colours as a tone of its own
(`render/shaders/lightsimulation.frag`).

### The pose results now

What the rows of Part 3 draw under the line math, one test per row (`tests/test_show_layers.py`,
at the 14° interval). Where a row draws other than Part 3 says, the difference is a consequence of
the additive filter: the overtone's phase against the fundamental changes the wave's shape, and
under the level a shape is a thickness.

| Pose                                        | Draws now                                                                   |
|---------------------------------------------|-----------------------------------------------------------------------------|
| neutral                                     | full blue over the window: the blue ping                                    |
| raised                                      | full white over the window, no blue: the bass                               |
| arms out level, a T                         | white lines a third of the interval, blue lines between; no sub-line yet    |
| left arm up, right hanging                  | white lines half the interval, one per interval; blue sub-lines             |
| right arm up, left hanging                  | white sub-lines a quarter of the interval, two per interval; blue lines     |
| a T, both elbows folded                     | the lines moved out half an interval, and two thirds wide                   |
| a T, left elbow folded                      | the lines moved out half an interval, a third wide                          |
| a T, right elbow folded                     | the lines in place, two thirds wide                                         |
| a T, leaning                                | the interval doubled or halved at full lean, back when upright              |
| a T, in a crouch                            | the blue's interval past the white's by the detune: beating                 |

At half registration the sub-line is a point: the overtone's crest between the lines reaches the
level exactly and lights nothing. It appears as the arms rise above level, and grows with the
harmonic's drawbar past the fundamental's.

### The dummy

The dummy (`pose/dummy.py`, settings `PI.dummy`) stands in for a person while the instrument is
judged: a figure of the pipeline's 17 landmarks whose joints are set by the panel, each joint's
degrees the angle the angle extractor reads at it when the figure is upright, the signed angle
from the segment above to the limb below, the right side mirrored as the extractor mirrors it
(the shoulder 0 hanging, 90 across the body, 180 up, 270 out; the elbow 180 straight, 0 folded;
the hip 180 standing, 90 leg out level; the knee 180 straight, 90 bent), standing at `azimuth`.
What is set is what the extractor reads. The `torso` leans the upper body over standing legs, as
a person leans: the arms, measured against the leaning torso line, still read as set; the hips
read off neutral by the lean, as a person's do. Its sides are named as the pipeline names
people's, the left on image-right.
Its frame joins the interpolated
poses before the LERP filters at its own id, `max_players`, between the live players and the
ghosts (which start one above it), so it is a pose like any other from there: the filters stamp
the playhead offset, the symmetries, the leg deviation and the bend on it, the render draws its
figure over the projection row where it stands (as it does every pose, `render.pose_figures`)
and its skeleton with the same data graphs as a player's in the pose row's last column,
the azimuth overlay marks it, Max plays it at its slot and the instrument lights it. Only the
joints are set; the leg deviation and the bend are the pipeline's, derived from the figure as for
a person, and its angles pass through the angle extractor and the calibrator as a person's do.
What the pipeline reads of its poses is what the calibrator is read against: the preset's
calibration is the dummy's raw readings, so its hanging arm reads 0 and its vertical arm π.

Its poses are named in `data/poses.json`, the rows of *Pose results*, picked in the `pose`
select and saved under `name` with `save`. A change of any measure morphs over `morph` seconds
(`easeInOutSine`), the azimuth and the arms the shortest way round, an exact opposite going up
through the front on both sides. `enabled` is forced off at startup; in the show the dummy is
never in the room.

---

## Open

Parts 1 to 3 are revisited after the first build runs on the machine; until then the theory stands
as written and the pose results are what the build is measured against.

Meaning:

- The elbows at the fixed points (*Pose results*): with the arms hanging the white is silent, so a
  folded elbow shows nothing; whether that is right
- The lean moves the hips' reading by the lean, on the dummy as on a person, so at full lean the
  leg deviation reads about 0.75 and the detune sounds with the bend; whether the leg deviation
  should be taken against the vertical instead of the torso line
- The symmetries: which of the four pairs earns a connection, and what it means
- Drift's source: a setting for now; whether a secondary measure should take it
- The composition itself: which connections people find with their bodies

Vocabulary not yet in the design:

- Tuning: intervals from a scale (4°, 6°, 8°, 12°, 24°, the series of the 90-line limit), so that
  synced people sit in simple ratios and their union reads as a chord; a scale means steps, and
  portamento between them is motion of its own
- Oscillator sync between people: one pattern's grid adopting the other's where they meet, an
  alternative to the union
- Expression: the motion features as sources for the push and the tint
- XOR as a second way for the colours to meet
- Decay and sustain on the window: the arrival flourish

Design:

- The tint between 0 and 1 is the central fraction of each line; whether that reads, or the
  line should take the other colour whole past some blend

From the machine **(site facts)**:

- The mask is too narrow and the window too wide at the preset's values
- Whether a line appearing at the limit width flickers when a pose hovers at its threshold
  **(deduction: the pipeline's smoothing should hold it)**
- Whether the 90-line limit holds with the moiré of two synced patterns
- A pitched wide-FOV camera reads a small spurious body bend near the frame edges, a pitch bend
  nobody played; check a straight person at the edge
