"""The sphere of screens and how the players' cells are wired to it: the one place that knows which
screen a cell is. Used by the laughter test (the number a lit cell shows), the scrolling name (the
screens of a row) and `data/audio/screen_angles.txt` (the angle of every cell, for the laughter).

The sphere has 12 ribs 30 degrees apart (rib 0 at the camera, counting to the LEFT seen from above
with the camera's right positive, the Max patch's convention: a visitor facing the sphere sees them
count to the right; checked on site 2026-10-25),
holding 5 and 3 screens by turn at heights 0 (top) to 4; a 3-rib holds the heights 1, 2 and 3. The
screens are numbered from 1 down each rib, then the next rib: 1-5 the first 5-rib, 6-8 the 3-rib
after it, 9-13, 14-16 and so on round. The wiring: screen s is Datapath unit ceil(s / 4), output
((s - 1) mod 4) + 1; player 1's units PL1-PL6 carry screens 1-24, player 2's 25-48; unit n is column
n - 1 of the player's 6 x 4 grid and output k its row k - 1, so cell = row * 6 + column. That puts the
heights 1, 2 and 3 of a player's six ribs on its grid rows 1, 2 and 3 in order, and the tops and
bottoms of its three 5-ribs by turns on grid row 0."""

RIBS: int = 12
RIB_STEP_DEG: float = -30.0             # per rib, seen from above with the right positive: the ribs count to the left
HEIGHTS: int = 5                        # rows of the sphere, 0 the top
RIB_HEIGHTS: tuple[range, range] = (range(0, 5), range(1, 4))   # the heights a 5-rib and a 3-rib hold
PAIR_SCREENS: int = 8                   # a 5-rib and the 3-rib after it
SCREENS: int = RIBS // 2 * PAIR_SCREENS  # 48
PLAYERS: int = 2
CELLS: int = SCREENS // PLAYERS         # 24 per player
COLUMNS: int = 6                        # the player's grid: Datapath units
OUTPUTS: int = 4                        # grid rows: a unit's outputs


def rib_height(screen: int) -> tuple[int, int]:
    """The rib (0 at the camera) and height (0 the top) of screen `screen` (1 to 48)."""
    pair, k = divmod(screen - 1, PAIR_SCREENS)
    if k < 5:
        return 2 * pair, k
    return 2 * pair + 1, k - 4


def screen_at(rib: int, height: int) -> int | None:
    """The screen (from 1) at `rib` and `height`, None where the rib has no screen at that height."""
    rib %= RIBS
    pair, odd = divmod(rib, 2)
    if height not in RIB_HEIGHTS[odd]:
        return None
    return pair * PAIR_SCREENS + 1 + (height if not odd else 5 + height - 1)


def row_screens(height: int) -> list[int]:
    """The screens (from 1) at `height` in order round the sphere: 12 for heights 1 to 3, 6 for 0 and 4."""
    return [s for rib in range(RIBS) for s in [screen_at(rib, height)] if s is not None]


def screen_of_cell(player: int, cell: int) -> int:
    """The screen (from 1) a cell (0 to 23) of player 0 or 1 shows, by the wiring."""
    row, column = divmod(cell, COLUMNS)
    return player * CELLS + column * OUTPUTS + row + 1


def cell_of_screen(screen: int) -> tuple[int, int]:
    """The player (0 or 1) and its cell (0 to 23) that show screen `screen` (1 to 48)."""
    player, rest = divmod(screen - 1, CELLS)
    column, row = divmod(rest, OUTPUTS)
    return player, row * COLUMNS + column


def rib_angle(rib: int) -> float:
    """The rib's angle seen from above: 0 at the camera, positive to the right, in -150 to 180."""
    angle: float = (rib * RIB_STEP_DEG) % 360.0
    return angle - 360.0 if angle > 180.0 else angle


def cell_angles() -> list[float]:
    """The angle of every cell, indexed as the Max patch does: cell + 24 for player 2 (the lines of
    data/audio/screen_angles.txt)."""
    return [rib_angle(rib_height(screen_of_cell(player, cell))[0]) for player in range(PLAYERS) for cell in range(CELLS)]


def screen_angles_text() -> str:
    """The contents of data/audio/screen_angles.txt (a coll: `index, angle;` per line, plain ASCII)."""
    return ''.join(f'{index}, {int(angle)}.;\n' for index, angle in enumerate(cell_angles()))
