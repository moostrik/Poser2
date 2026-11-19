---
description: Naming and measurements in the flow
paths:
  - "modules/flow/**"
  - "modules/render/modules/flow/**"
  - "apps/hd_trio/flow/**"
  - "apps/hd_trio/render/layers/flow/**"
  - "apps/deep_flow/render/layers/view/**"
  - "apps/low_flow/render/layers/view/**"
---
# Flow naming

- Name the fluid's things by the textbook word of fluid simulation where one exists: `wall` for the domain's edge, `lowpass` and `bandpass` for the filters; boundary (the body in the water) and body force (the Navier–Stokes force term) say how an input goes in, in its description, not its name
- Name a reading by what it is (`trail`, `change`, `start`), and a module by what it does (`VelocityLowpass`, `VelocityBandpass`)
- Name a value by its quantity (`velocity`, `acceleration`); the words "impulse", "body", "force" and "delta" name nothing on their own
- Prefer the flow's textbook units: widths, widths/s, widths/s², seconds
- A setting's default is the textbook value

# Flow measurements

- Before a measurement, state what it is to show and which value shows it; ask when that is not clear
- A measurement is a time (what a pass costs, in ms) or a result (what a stage makes: noise, real z, fit, the look)
- A measured table names its kind, date, commit, recording and settings; a result also its runs and their spread
- Ask to measure a time again when its pass's code or size changes
- Compare a result only with rows of the same recording, settings and code paths
- Once a row is written, ask to delete its captures and replays
