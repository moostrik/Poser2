# White Space — Arm Matrix v2

The arms' connections to the light synth, rebuilt from the base one connection at a time.
Parameters are the synth's (`LIGHT_SYNTH.md`); each row's value is `base + amount × curve(source)`.
`MATRIX.md` is the first build's record, Options 1–4.

## Rules

The rules are ranked, as the laws of robotics are: a rule holds except where it would break a rule
above it.

1. Neutral (arms hanging) is full blue.
2. Raised (both arms up) is full white.
3. No jumps: a small movement of an arm is a small change of the light.
4. Keep the visualisation as simple as possible: a connection must earn its place. Temporary,
   while the matrix is rebuilt, because the audience has to find their own representation first
   **(site fact)**; above rule 5, since a rule below it could not excuse any simplification.
   When it goes, rule 5 moves back up.
5. Every combination of the arms draws differently.

## Measures

What the pose pipeline says about the body, smoothed and through its dead zones
(`POSE_INSTRUMENT.md`, *The body*); listed whether or not a row uses it.

| Measure           | Range | What it is                                                    |
|-------------------|-------|---------------------------------------------------------------|
| left shoulder     | 0..1  | 0 hanging → 1 straight up                                     |
| right shoulder    | 0..1  | 0 hanging → 1 straight up                                     |
| left elbow        | 0..1  | 0 straight → 1 folded                                         |
| right elbow       | 0..1  | 0 straight → 1 folded                                         |
| left elbow angle  | −π..π | the raw angle, 0 straight → ±π folded, the sign the side      |
| right elbow angle | −π..π | the raw angle, as the left's                                  |
| body bend         | −1..1 | −1 leaning left → 0 upright → 1 leaning right                 |
| leg deviation     | 0..1  | 0 standing straight → 1 a leg fully bent                      |
| distance          | 0..1  | 0 the zone's near edge → 1 its far edge                       |
| symmetry          | −1..1 | per pair, left minus right (`POSE_INSTRUMENT.md`, *Symmetry*) |

The similarity is not a matrix source: it drives sync, the reach toward a partner
(`POSE_INSTRUMENT.md`, *Events*).

## Made sources

What the instrument makes or shapes from the measures before the slot: a mean and a difference
of two, a sine of an angle, a sine in time. A slot has one source and its curve keeps 0 and ±1 in place
(`LIGHT_SYNTH.md`, *Modulation*), so none of these can be made in the slot. They are computed in
the instrument while the matrix is tried; a shaped measure moves into the pipeline once liked
(`POSE_INSTRUMENT.md`, *The rules of meaning*).

| Source              | Range | What it is                                                      |
|---------------------|-------|-----------------------------------------------------------------|
| shoulders           | 0..1  | the mean of the two shoulders                                   |
| shoulder difference | −1..1 | the left shoulder less the right, signed                        |
| left elbow turn     | −1..1 | sine of the elbow's angle: ±1 at ±90°, 0 straight or folded     |
| right elbow turn    | −1..1 | sine of the elbow's angle                                       |
| breath              | −1..1 | a sine in time at `PI.breath.rate`, one per person              |

## Synth inputs

The slots a row can fill, per voice; what each parameter looks like is `LIGHT_SYNTH.md`'s
(*Parameters*, *The strobe*, *Modulation*).

| Slot                         | Unit                 |
|------------------------------|----------------------|
| pitch, white and blue        | lines per revolution |
| pulse width, white and blue  | fraction of interval |
| phase, white and blue        | intervals            |
| speed, white and blue        | degrees per second   |
| hardness, white and blue     | 0..1                 |
| strobe rate, white and blue  | strobes per second   |
| strobe width, white and blue | fraction of a cycle  |
| strobe phase, white and blue | cycles               |
| strobe shift, white and blue | half cycles per line |
| the LFO's level              | 0..1                 |

Not rows of the matrix: the push and its release (an envelope on the hit, `LIGHT_SYNTH.md`,
*Distance and time*), the window's reaches and presence (*The window*), and the switches
(enabled, mirror, the bypasses). The events that drive them are `POSE_INSTRUMENT.md`'s
(*Events*).

## Base matrix

The shoulders' mean into both pulse widths, nothing else. The bases are the studio preset's.

| Oscillator | Parameter   | Source    | Base | Amount |
|------------|-------------|-----------|------|--------|
| white      | pitch       |           | 10   |        |
| white      | pulse width | shoulders | 0    | 1      |
| white      | phase       |           | 0    |        |
| white      | speed       |           | 0    |        |
| white      | hardness    |           | 1    |        |
| blue       | pitch       |           | 10   |        |
| blue       | pulse width | shoulders | 1    | −1     |
| blue       | phase       |           | ½    |        |
| blue       | speed       |           | 0    |        |
| blue       | hardness    |           | 1    |        |

Hanging is full blue and raised full white (rules 1 and 2); between them the two widths sum to
the interval and blue sits half an interval from white, so the colours tile: every pixel is one
colour, trading blue for white as the shoulders rise. The lines stand still, so a held pose is a
still picture and every movement of the light is an arm moving. The strobes and the push have no
source; the legs feed the LFO's level, but the LFO feeds no parameter. Rule 5 yields to rule 4:
the elbows, the turns, the bend and the legs draw nothing, and which arm is up does not show.
Each connection added wins rule 5 back one combination at a time, at the price rule 4 allows.

## Step 1 — the elbows

The base matrix with each elbow on its own colour's pitch, the left the white and the right the
blue.

| Oscillator | Parameter   | Source      | Base | Amount |
|------------|-------------|-------------|------|--------|
| white      | pitch       | left elbow  | 10   | 20     |
| white      | pulse width | shoulders   | 0    | 1      |
| white      | phase       |             | 0    |        |
| white      | speed       |             | 0    |        |
| white      | hardness    |             | 1    |        |
| blue       | pitch       | right elbow | 10   | 20     |
| blue       | pulse width | shoulders   | 1    | −1     |
| blue       | phase       |             | ½    |        |
| blue       | speed       |             | 0    |        |
| blue       | hardness    |             | 1    |        |

Folding an elbow makes its own colour's lines finer: a full fold triples them, 10 (one every
36°) to 30 (one every 12°). The first build rested at 25.7 and folded to 72 (one every 5°), and
those small lines lost the audience **(site fact)**: this step rests far coarser, and even
folded the lines stay lines. Straight against folded is 1:3, a simple ratio, so the unequal
elbows' moiré still repeats regularly along the wall. The relation between the arms is tuning: equal elbows keep white and blue sharing one
interval, in tune; unequal elbows slide the colours past each other with distance, a standing
moiré (`LIGHT_SYNTH.md`, *In the pose instrument*), so the elbows' symmetry shows without a
connection of its own.

The step keeps every rule above it for free. The elbow travel is 0 at both fixed points — hanging
and raised arms are straight — and a pitch shows nothing on a solid or dark colour, so rules 1
and 2 hold twice over. The travel is continuous and a change of pitch opens the lines from the
person as an accordion, so rule 3 holds. Nothing moves in time: a held pose is still a still
picture. The price to rule 4 is two rows, one measure each; the gain to rule 5 is every elbow
combination, wherever the shoulders give lines to see it in.

## Step 2 — the shoulder difference

The wired matrix (`PoseInstrument.connect`): Step 1 with the shoulders' difference on both
phases, the amounts opposite.

| Oscillator | Parameter   | Source              | Base | Amount |
|------------|-------------|---------------------|------|--------|
| white      | pitch       | left elbow          | 10   | 20     |
| white      | pulse width | shoulders           | 0    | 1      |
| white      | phase       | shoulder difference | 0    | ⅛      |
| white      | speed       |                     | 0    |        |
| white      | hardness    |                     | 1    |        |
| blue       | pitch       | right elbow         | 10   | 20     |
| blue       | pulse width | shoulders           | 1    | −1     |
| blue       | phase       | shoulder difference | ½    | −⅛     |
| blue       | speed       |                     | 0    |        |
| blue       | hardness    |                     | 1    |        |

The difference moves the two colours' lines opposite ways: level shoulders leave them
alternating, blue half an interval from white; the left higher brings them a quarter interval
closer on one side, the right higher on the other, so which arm is the higher shows — rule 5
wins the shoulder pairs back. ⅛ each keeps those two furthest apart: at ¼ each, left higher and
right higher would draw the same lines, as a phase repeats every interval. Pulled together the
colours overlap on one flank and open dark on the other, so the shift reads as a change of tone,
not only of place.

The difference is 0 with level shoulders, so the fixed points, the T and the tiling are
untouched, and it is large only with one arm up and the other not, where both widths are near
half and both colours are lines. Nothing moves in time, and the phase is continuous in the
difference, so rules 1, 2 and 3 hold.
