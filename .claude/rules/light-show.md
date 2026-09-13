---
description: White Space light layers and state machine — their inputs
paths:
  - "apps/white_space/light/**"
  - "apps/white_space/statemachine/**"
---
# Light show inputs

- Read people from pose frames on the board (`get_frames`, `get_ghosts`), never from the tracker's tracklets or observations
- Treat presence as the pose itself; a layer adds no presence test of its own
