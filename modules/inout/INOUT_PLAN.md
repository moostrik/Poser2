# Inout plan

`inout` is the network output and input the apps share: OSC out (`OscSender`, `OscSound`), OSC in (`OscReceiver`)
and Art-Net out (`ArtNetSender`, `ArtNetBars`). The aim is one OSC system for every app that sends or receives
OSC: hd_trio, deep_flow, low_flow, white_space and smile. hd_trio's output stays byte for byte as it is.

Checks run in `.venv-oak4` (inout, deep_flow, low_flow, smile, pyright), except white_space's and hd_trio's tests,
which need depthai 2 and run in `.venv`.

## B. OSC receiver

- B1. Callbacks, teardown and port change. Why: `OscReceiver._handle` keeps callbacks in a plain list and catches
  only `TypeError`, so any other error skips the address's later callbacks; `__init__` binds `ip_out`, `port_out`
  and `port_in` and nothing unbinds them; `_on_in_change` swaps `self.server` from another thread without a lock
  while `bind` reads it (the race is a deduction); `last_time` is stamped with `time()`. How: a `utils.Broadcast`
  per address, bind at `start` and unbind at `stop`, the swap under a lock, `perf_counter()` for the stamp and
  `last_time`'s description to match. Check: a loopback test: a bound callback gets its arguments, a failing
  callback does not stop the next, a port change keeps the bindings.
- B2. `OscReceiverSettings` fields typed. Why: its fields have no `Field[T]` annotation, unlike every other settings
  class in the package. How: annotate them. Check: pyright.

## C. Art-Net

- C1. `ArtNetSender` for both apps: universes, a flash switch, sends on a change, blackouts. Why: hd_trio's bars send one packet to two
  consecutive universes, and their stop is a blackout only, while `ArtNetSender` sends one universe and flashes at
  enable, address change and stop; disabled, it only stops sending, so the lights hold their last frame, and it
  sends no blackout at start **(seen)**; it sends every tick, `ArtNetBars` on a change and once a second. How: the
  count of consecutive universes and a flash switch through the constructor; one StupidArtnet per universe; one
  sending mode for both apps, as Art-Net 4 has it (art-net.org.uk/downloads/art-net.pdf): a universe goes out when
  its packet's bytes differ from the last sent, and as a keep-alive about once a second (the spec's revision notes:
  800–1000 ms), at most 44 Hz; a blackout at start, on disable and at stop, after the flash where the flash is on,
  for both apps. smile keeps one universe and the flash, sends less while its lights hold still, and no longer
  holds its last frame when switched off. `ArtNetSender` owns its thread instead of subclassing `Thread`, its caller
  (smile) and tests updated. Check:
  `test_artnet_sender` (a send on a change, the keep-alive, the 44 Hz cap, the blackouts at start, disable and
  stop, two universes); smile's tests updated.
- C2. hd_trio's bars as a fill on `ArtNetSender`. Why: `ArtNetBars` and `ArtNetSender` each own a StupidArtnet, a
  thread and a tick; `ArtNetBars` is hd_trio's bar look, used only by hd_trio. Its own faults go with it:
  `_update_packet_size`, `_update_universe`, `_reinit_channel_order`, `_validate_and_reinit_artnet` and
  `_reinit_artnet` have no caller, so a panel change of `ip`, `pixels`, `base_universe`, `channel_order` or `fps`
  does nothing until a restart; `_update_loop` returns for good when `validate_connection` fails at start and on a
  `socket.error`; `__del__` tears down from the garbage collector, and never runs while the settings live, since
  their callback lists hold its bound methods, while `CLAUDE.md` says to clean up in an explicit stop or teardown,
  never in `__del__`. How: the bar fill (`ArtNetBarsSettings`' look fields, `ChannelOrder`, the fill) moves to
  `apps/hd_trio` on `ArtNetSender` with two universes and the flash off; the bars keep their blackout at start, at
  stop and when switched off; the `bar` field keeps whatever sets it today (nothing in hd_trio's code does: the
  panel or a share **(seen)**); `ArtNetBars` goes; hd_trio's settings files migrated. Check: `test_artnet_bars`
  moved to `apps/hd_trio/tests`.

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

- `CLAUDE.md` Concurrency, a callback list is a `utils.Broadcast`; a failing callback does not stop the others: B1
- `CLAUDE.md` Error handling, a thread's loop catches an exception per item and goes on: C2
- `CLAUDE.md` Settings, bind() / unbind(): B1
- `CLAUDE.md` Composition, prefer composition over inheritance: C1
- `CLAUDE.md` Composition and wiring, clean up in an explicit stop or teardown, never in `__del__`: C2
- `CLAUDE.md` Time, `perf_counter()` for every stamp: B1
- `CLAUDE.md` API evolution, remove obsolete pathways; keep settings files in sync: C2, E1
- `CLAUDE.md` Types, modern typing: B2
- `CLAUDE.md` Project shape, code only one app uses moves to that app: C2
- `CLAUDE.md` Threads, the finder fixes a minor problem and asks the owning thread for feedback: E1
- `.claude/rules/docs.md`, a component is described in its own document: D1