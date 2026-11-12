---
description: Naming in the flow: the fluid's things by the textbook word, readings and modules
paths:
  - "modules/flow/**"
  - "modules/render/layers/flow/**"
---
# Flow naming

- Name the fluid's things by the textbook word of fluid simulation where one exists: `boundary` for the body in the water, `wall` for the domain's edge, `body_force` for the Navier–Stokes force term, `lowpass` and `bandpass` for the filters
- Name a reading by what it is (`trail`, `change`, `start`), and a module by what it does (`VelocityLowpass`, `VelocityBandpass`)
- Name a value by its quantity (`velocity`, `acceleration`); the words "impulse", "body", "force", "delta" and "drag" name nothing on their own
- Give every quantity and setting a unit that means something on its own (widths/s, seconds, 0 clear to 1 fully dyed, ±1 warm); choose a setting's value for what its unit says, never as a factor derived from an old look or another setting, and fix the unit where the quantity is made when a value would need one
- One quantity has one gain
- A setting's default is the textbook value
