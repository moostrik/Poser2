---
description: Panels and monitors: settings mirror the code, a monitor shows the structure
paths:
  - "apps/deep_flow/**"
  - "modules/render/modules/flow/**"
  - "modules/flow/**"
---
# Panels

- A settings class mirrors the code it configures: a group for each part, in the order the code runs them
- Within a group, prefer the fields in the order the code uses them
- A component's monitors come last in its settings
- A panel that holds more than one chain (an app's root, the render panel) may order its groups by use
- A stage's ports follow its parts: each part's textures under its name, in the order the stage runs them, so a monitor shows the structure as well as the picture
- When a settings path or a port name reads wrong, fix the structure it comes from, not the name
