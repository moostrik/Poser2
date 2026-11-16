---
description: Naming in the flow: the fluid's things by the textbook word, readings and modules
paths:
  - "modules/flow/**"
  - "modules/render/modules/flow/**"
  - "apps/hd_trio/flow/**"
  - "apps/hd_trio/render/layers/flow/**"
---
# Flow naming

- Name the fluid's things by the textbook word of fluid simulation where one exists: `boundary` for the body in the water, `wall` for the domain's edge, `body_force` for the Navier–Stokes force term, `lowpass` and `bandpass` for the filters
- Name a reading by what it is (`trail`, `change`, `start`), and a module by what it does (`VelocityLowpass`, `VelocityBandpass`)
- Name a value by its quantity (`velocity`, `acceleration`); the words "impulse", "body", "force" and "delta" name nothing on their own
- Prefer the flow's textbook units: widths, widths/s, widths/s², seconds
- A setting's default is the textbook value
