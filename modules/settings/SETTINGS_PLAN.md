# Settings plan

## Parked

- hd_trio's render writes `settings.fluid.density.fade_time` every tick from the motion
  (`apps/hd_trio/render/hd_trio_render.py`): a component writing settings it does not own. Taken up with hd_trio.

## Rules this plan touches

- CLAUDE.md: Settings connect the user and a component, not two components
