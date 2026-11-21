# Inout plan

`inout` is the network output and input the apps share: OSC out (`OscSender`, `OscSound`), OSC in (`OscReceiver`)
and Art-Net out (`ArtNetSender`). The aim is one OSC system for every app that sends or receives
OSC: hd_trio, deep_flow, low_flow, white_space and smile. hd_trio's output stays byte for byte as it is.

Checks run in `.venv-oak4` (inout, deep_flow, low_flow, smile, pyright), except white_space's and hd_trio's tests,
which need depthai 2 and run in `.venv`.

## D. Document

- D1. `INOUT.md`: the senders and receivers, which app uses which, and the OSC addresses each one puts on the wire
  (the contract with the Max patches), now spread over docstrings and comments. Why: the package has no document.

## E. Apps

- E1. smile's walls' bang senders in the settings tree. Why: `Player.__init__` (`apps/smile/player/player.py`)
  builds them on `OscSenderSettings()` outside the settings tree (ports 4115 and 4116, ip 127.0.0.1), so no panel
  shows them and no settings file holds them. How: their ip and ports as fields in the walls' settings, smile's
  settings files migrated; built here, since smile's own part is done, and the smile thread is asked for feedback
  afterwards. Check: smile's tests.

## Stays in the apps

- white_space `OscLightSender`: the fixture firmware's wire contract (1220-byte datagrams, `/WS/blue2` last, paced
  chunks) does not fit a bundle sender, and the firmware never changes.
- white_space `UdpLightReceiver`: plain UDP text, not OSC.

## Rules this plan touches

- `CLAUDE.md` API evolution, keep settings files in sync: E1
- `CLAUDE.md` Threads, the finder fixes a minor problem and asks the owning thread for feedback: E1
- `.claude/rules/docs.md`, a component is described in its own document: D1