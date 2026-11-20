# Settings plan

## D. The package held up to the rules

- D6. `BaseSettings.__setattr__` looks for a `property` setter through the class's MRO before it looks in its own
  fields, on every write (SETTINGS.md, Measured); no settings class has a property setter. A name in `_fields` goes
  straight to `Field.set`; another name goes through `object.__setattr__` when the class declares it (a property, a
  `Group`, whose `__set__` refuses a replacement), else raises "has no setting". Checked by settings' tests.

## Parked

- hd_trio's render writes `settings.fluid.density.fade_time` every tick from the motion
  (`apps/hd_trio/render/hd_trio_render.py`): a component writing settings it does not own. Taken up with hd_trio.

## Rules this plan touches

- CLAUDE.md: Prefer the design with the lower latency, measured in the context of the app
- CLAUDE.md: Settings connect the user and a component, not two components
