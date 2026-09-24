# White Space — Arm Matrix

The arms' connections to the light synth, filled step by step. Parameters are the synth's
(`LIGHT_SYNTH.md`); each row's value is `base + amount × curve(source)`.

## Rules

The rules are ranked, as the laws of robotics are: a rule holds except where it would break a rule
above it.

1. Neutral (arms hanging) is full blue.
2. Raised (both arms up) is full white.
3. No jumps: a small movement of an arm is a small change of the light.
4. Every combination of the arms draws differently.

## Measures

Each 0..1 after the pipeline's dead zones (`POSE_INSTRUMENT.md`, *The body*): 0 at neutral; the
shoulders 1 at raised, the elbows 0 at raised.

| Measure             | What it is                                                              |
|---------------------|-------------------------------------------------------------------------|
| left shoulder       | 0 hanging → 1 straight up                                               |
| right shoulder      | 0 hanging → 1 straight up                                               |
| left elbow          | 0 straight → 1 folded                                                   |
| right elbow         | 0 straight → 1 folded                                                   |
| shoulders           | the mean of the two shoulders' absolute values                          |
| shoulder difference | the left shoulder's absolute value minus the right's, −1..1             |
| left elbow turn     | sine of the left elbow's signed angle: ±1 at ±90°, 0 straight or folded |
| right elbow turn    | sine of the right elbow's signed angle                                  |
| left excess         | how much higher the left shoulder is: the difference, 0 below 0         |
| right excess        | how much higher the right shoulder is: minus the difference, 0 below 0  |

Option 3 adds the breath, a source the instrument makes in time (*Option 3 Matrix*).

## Base matrix

| Oscillator | Parameter   | Source    | Base | Amount |
|------------|-------------|-----------|------|--------|
| white      | pitch       |           |      |        |
| white      | pulse width | shoulders | 0    | 1      |
| white      | phase       |           |      |        |
| white      | speed       |           |      |        |
| white      | hardness    |           | 1    |        |
| blue       | pitch       |           |      |        |
| blue       | pulse width | shoulders | 1    | −1     |
| blue       | phase       |           |      |        |
| blue       | speed       |           |      |        |
| blue       | hardness    |           | 1    |        |

## Option 1 Matrix

The base matrix with each elbow on its colour's pitch. Speed is not the arms'. The pitch's base
and amount are the preset's: 25.7 lines (one every 14°) straight, 72 (one every 5°) folded.

| Oscillator | Parameter   | Source              | Base | Amount |
|------------|-------------|---------------------|------|--------|
| white      | pitch       | left elbow          | 25.7 | 46.3   |
| white      | pulse width | shoulders           | 0    | 1      |
| white      | phase       | shoulder difference | 0    | ⅛      |
| white      | speed       |                     |      |        |
| white      | hardness    |                     | 1    |        |
| blue       | pitch       | right elbow         | 25.7 | 46.3   |
| blue       | pulse width | shoulders           | 1    | −1     |
| blue       | phase       | shoulder difference | ½    | −⅛     |
| blue       | speed       |                     |      |        |
| blue       | hardness    |                     | 1    |        |

Folding an elbow makes its own colour's lines finer; equal elbows keep white and blue in tune,
unequal ones slide the colours apart. With both arms hanging or both raised the colours are
solid and dark, so a folded elbow does not show there: rules 1 and 2 rank above rule 4.

The shoulder difference moves the two colours' lines opposite ways: level shoulders leave them
alternating, blue half an interval from white; the left higher brings them a quarter interval
closer on one side, the right higher on the other. ⅛ each keeps those two furthest apart: at ¼
each, left higher and right higher would draw the same lines, as a phase repeats every interval.
The difference is large only with one arm up and the other not, where both colours are lines.

## Option 2 Matrix

Option 1 with each elbow's turn on its colour's speed: the elbow bent +90° flows its colour one
way, −90° the other, straight and fully folded leave it at its base. The speed's base and amount
are the preset's.

| Oscillator | Parameter   | Source              | Base    | Amount |
|------------|-------------|---------------------|---------|--------|
| white      | pitch       | left elbow          | 25.7    | 46.3   |
| white      | pulse width | shoulders           | 0       | 1      |
| white      | phase       | shoulder difference | 0       | ⅛      |
| white      | speed       | left elbow turn     | 3.9°/s  | 15°/s  |
| white      | hardness    |                     | 1       |        |
| blue       | pitch       | right elbow         | 25.7    | 46.3   |
| blue       | pulse width | shoulders           | 1       | −1     |
| blue       | phase       | shoulder difference | ½       | −⅛     |
| blue       | speed       | right elbow turn    | −4.5°/s | 15°/s  |
| blue       | hardness    |                     | 1       |        |

The turn is a sine so that the flow is still at full fold: there +180° and −180° are one pose,
and a turn read straight from the angle would reverse the flow at once as the elbow crosses it.
Each elbow plays its own colour whole: how fine its lines are and which way they flow. Unequal
turns set the colours flowing at their own speeds, apart or opposite. The sign of the turn is the
side the forearm folds to, which the pitch, on the absolute angle, does not see; the extractor
mirrors the right arm, so both forearms folded alike flow both colours alike, and which fold flows
outward is the sign of the amount.

## Option 3 Matrix

Option 2 with a breathing width. The instrument makes the breath as part of the source it hands
to each pulse width slot, as it makes the mask's
flash: the shoulders' mean sets where the width rests, and the higher shoulder makes its own
colour's width swing around it. The synth is unchanged. The phases go: they showed the shoulder
difference only while moving.

| Source      | What it is                                                               |
|-------------|--------------------------------------------------------------------------|
| breath      | a sine in time, −1..1, at the breath's rate (0.5 Hz), one per person     |
| white width | shoulders + depth × left excess × breath                                 |
| blue width  | shoulders + depth × right excess × breath                                |
| white flow  | left elbow turn + body bend                                              |
| blue flow   | right elbow turn + body bend                                             |

The depth is a setting, 0.4; at most ½ (below).

| Oscillator | Parameter   | Source           | Base    | Amount |
|------------|-------------|------------------|---------|--------|
| white      | pitch       | left elbow       | 25.7    | 46.3   |
| white      | pulse width | white width      | 0       | 1      |
| white      | phase       |                  | 0       |        |
| white      | speed       | white flow       | 3.9°/s  | 15°/s  |
| white      | hardness    |                  | 1       |        |
| blue       | pitch       | right elbow      | 25.7    | 46.3   |
| blue       | pulse width | blue width       | 1       | −1     |
| blue       | phase       |                  | ½       |        |
| blue       | speed       | blue flow        | −4.5°/s | 15°/s  |
| blue       | hardness    |                  | 1       |        |

One breath serves both colours: only one shoulder can be the higher, so only one colour breathes
at a time. The body bend is not the arms', but shares the speed slots with the elbow turns: white
drifts outward and blue inward, so a bend added to both with one sign makes the white faster and
the blue slower one way, the reverse the other. A turn and a lean share the amount, so they add
or cancel.

| Pose                   | White            | Blue             |
|------------------------|------------------|------------------|
| neutral                | none, still      | full, still      |
| raised                 | full, still      | none, still      |
| T                      | ½, still         | ½, still         |
| left up, right hanging | ½, breathing     | ½, still         |
| right up, left hanging | ½, still         | ½, breathing     |

Level shoulders have no excess, so both widths are still at neutral, at raised and in a T. One
shoulder can only be much higher than the other where the mean is well inside 0..1: the excess is
at most twice the smaller of the mean and one less the mean, so with a depth of ½ or less a breath
never passes full or none, and the width reaches both fixed points exactly. The breathing colour
no longer tiles with the still one: as it swells past the other's gap the overlap tone shows, as
it thins, dark.

## Option 4 Matrix

The wired option (`PoseInstrument.connect`): Option 3 with the breath symmetric, the turns on the
phases and the speed the body bend's alone.

Both colours breathe, as far as the shoulders are apart; which shoulder is the higher sets the
relation between them, not which one breathes. One colour breathing while the other stood still
read as one arm doing much more than the other, since white breathing shows far stronger than
blue.

| Source      | What it is                                                               |
|-------------|--------------------------------------------------------------------------|
| breath      | a sine in time, −1..1, at the breath's rate (0.5 Hz), one per person     |
| gap         | how far apart the shoulders are: the difference, unsigned                |
| swing       | depth × gap × breath                                                     |
| white width | shoulders + swing                                                        |
| blue width  | shoulders − swing with the left higher, shoulders + swing with the right |

The depth is a setting, 0.4; at most ½ (below).

| Oscillator | Parameter   | Source           | Base   | Amount |
|------------|-------------|------------------|--------|--------|
| white      | pitch       | left elbow       | 25.7   | 46.3   |
| white      | pulse width | white width      | 0      | 1      |
| white      | phase       | left elbow turn  | 0      | ¼      |
| white      | speed       | body bend        | 0°/s   | 15°/s  |
| white      | hardness    |                  | 1      |        |
| blue       | pitch       | right elbow      | 25.7   | 46.3   |
| blue       | pulse width | blue width       | 1      | −1     |
| blue       | phase       | right elbow turn | ½      | ¼      |
| blue       | speed       | body bend        | 0°/s   | 15°/s  |
| blue       | hardness    |                  | 1      |        |

| Pose                   | White               | Blue                          |
|------------------------|---------------------|-------------------------------|
| neutral                | none, still         | full, still                   |
| raised                 | full, still         | none, still                   |
| T                      | ½, still            | ½, still                      |
| left up, right hanging | ½, breathing        | ½, breathing in step          |
| right up, left hanging | ½, breathing        | ½, breathing complementary    |

Blue's slot inverts its source, so the same swing on both is the two colours breathing together,
between dark and the overlap tone, and the opposite swing keeps them complementary, the boundary
between them sliding. The sign turns where the shoulders are level and the swing is 0 there, so
nothing jumps. Level shoulders have no gap, so neutral, raised and a T are still; the gap is at
most twice the smaller of the mean and one less the mean, so a depth of ½ or less never takes a
width past none or full and both fixed points stay exact.

The speed is the body bend's alone, its base 0: a straight body leaves the lines standing and
every movement of a still pose comes from moving the arms. A lean carries both colours the same
way, the amounts being the knobs if they should ever counter-flow. A turn shifts its own colour's
lines a quarter interval at ±90° and holds them where they are.

## Parked

- The body bend as a drift on the wall: both sides of the pattern flowing one way, outward on the
  side leaned to and inward on the other. Each mirrored oscillator then needs a travel per side,
  and a choice: after the lean the sides keep the offset they gained, or ease back into mirror.
- The body bend as the voice's time rate, a multiplier on every tick's time step: both speeds,
  their bases and the push scaled at once, the elbows still choosing each colour's direction.
  Signed, base 1 and amount 1: a lean one way slows the lines to a standstill, the other way
  doubles them. A new parameter of the synth (`LIGHT_SYNTH.md`, *Distance and time*: the clock).
