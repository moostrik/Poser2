# SMILE

Art installation. A camera watches the visitors; when a face holds still in front of it, the image
flashes and ComfyUI paints a portrait of that face, which then grins. The result is shown on the main
screen with music (a Max patch, driven over OSC) and lights (an Art-Net ring around the camera). Two
image players next to it show the grins of the visitors before.

Runs on one Windows PC: an RTX 4090 (24 GB) with 64 GB RAM, an Allied Vision Mako G-125C GigE camera
on the PC's Ethernet adapter, the show monitor and two player monitors, an Art-Net node for the
lights, and Max for the music. Everything the installation is given lives in `data\` (presets,
models, workflow, audio, test photos); everything it makes goes to `output\` (clips, portraits, logs;
not in git, safe to delete).

## INSTALLATION

### Visual C++ Redistributable
* install the latest Visual C++ Redistributable from [site](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170#latest-supported-redistributable-version) and RESTART

### Python
* install Python 3.12 Windows Installer (64 bit) [download](https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe) or [site](https://www.python.org/downloads/release/python-31210/)
* choose **Customize installation**, uncheck "Add python.exe to PATH" and keep the py launcher
* verify with ```py -0p``` that `-3.12` is listed

### Git for Windows
* install the latest version from [site](https://git-scm.com/download/win)
* configure the global user name and email
```git config --global user.name "M.Oostrik"```
```git config --global user.email "m.oostrik@gmail.com"```
* initialize Git LFS: ```git lfs install``` (the face models in `data\models\` and the fonts are LFS files)

### VSCode (optional, for development)
* install the latest version from [site](https://code.visualstudio.com/Download)
* extensions: Python, Python Environments, Python Indent, Pylance

### ExecutionPolicy
* open PowerShell as administrator and type ```Set-ExecutionPolicy -ExecutionPolicy RemoteSigned```, so that PowerShell scripts run

### Allied Vision camera
* download the Vimba SDK from [site](https://www.alliedvision.com/en/products/vimba-sdk/) (Vimba, not Vimba X: the app brings VimbaPython, which needs Vimba's VmbC)
* run the installer as administrator and restart
* the camera's `UserSet2` holds its base configuration (format, the region of interest 1200 x 960 centred on the sensor, the 4:5 of the display, frame rate, packet size); the app loads it at every open and writes the panel's camera settings over it; the log line at open shows the region and the packet size it got
* Ethernet adapter: Device Manager → the adapter → Advanced → **Jumbo Packet → 9014 Bytes** (it is off on a new PC and after a driver reinstall), and every switch between camera and PC must pass jumbo frames too. Otherwise the packet size stays 1500 and frames drop more easily. The SMILE log says at start which size the camera got: `packet size 8228` or more is good, `packet size 1500` means something on the path does not pass them and the log warns (the panel's `dropped_frames` shows how much that costs)

### PC (GIGABYTE Z590 AORUS MASTER, Windows 11)
* network adapter (onboard Marvell AQC107, the camera and everything else share it): driver 3.1.x (Microsoft Update Catalog, "Marvell Net Driver Update"; GIGABYTE's page has 3.1.3). With the 2.2.2.0 driver from 2020 the adapter hung with the link still up: camera and internet gone until the adapter was disabled and enabled (2026-10-05). Advanced tab: Jumbo Packet 9014 Bytes, Recv Segment Coalescing (IPv4 and IPv6) Disabled, Receive Buffers 4096; Power Management tab: nothing ticked
* power plan: PCI Express → Link State Power Management → Off; sleep, hibernate and display off: never
* Windows Update, in `gpedit.msc` under Windows Update → Manage updates offered from Windows Update: "Do not include drivers with Windows Updates" enabled, "Select the target Feature Update version" set to the installed version (25H2). Pause updates the week before a show
* BIOS: Settings → APP Center Download & Install Configuration → Disabled (otherwise the firmware writes a GIGABYTE update service into Windows at every boot). No GIGABYTE APP Center, no NVIDIA App, no Intel Driver & Support Assistant on the rig
* at login only the startup shortcut, the Realtek audio service and Windows Security start; the user logs in automatically

### ffmpeg
* download the "release full" build from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/), extract it to `C:\Developer\ffmpeg` and add `C:\Developer\ffmpeg\bin` to the PATH
* without it the visitors are not saved as clips and the players have nothing to play (`install.bat` checks)

### Install script
* in a terminal run ```scripts\install.bat```: pulls the LFS files, checks ffmpeg, makes the venv `.venv` and installs `requirements.txt`

### Image generator (ComfyUI)
* copy the ComfyUI bundle (with its models, ~23 GB) into the repo as `comfy\` and follow [comfy\README.md](comfy/README.md): `setup.ps1` installs ComfyUI in `ComfyUI\`; `comfy\models\` and `ComfyUI\` stay out of git
* SMILE runs the API export of the workflow in `data\comfy\smile_workflow.json` and fills in every value from its panel (re-export with **Export (API)** after changing the workflow in ComfyUI)
* `data\comfy\warmup.jpg` is the photo of the warm-up run SMILE does after every connect, so the models are loaded before the first visitor

### Music (Max)
* install Max (Cycling '74); the patch is `data\audio\Smile-Music-Player.maxpat`
* the music (`data\audio\SMILE-BS-Music\`) and the laughter samples (`data\audio\laughter\`) are in git LFS and come in with the models (`scripts\install.bat`, or `git lfs pull`); `data\audio\music_map.json` maps the two movements (`0000` for style A, `0001` for style B) to the tracks and their weights
* SMILE sends OSC to Max on `127.0.0.1:4114` (`/music_id`, `/state`, `/progress` and the controls below; the OSC card of the panel's Audio tab); the players send `/bang <cell>` on 4115 and 4116 when a clip starts in a cell
* the panel's Audio tab holds the Max panel's controls: `sound` (the patch's DSP), `music_volume` and `laughter_volume` (both dB), `laughter` (the samples those `/bang`s trigger) and `laughter_offset` (degrees, see the speakers below); SMILE sends them every frame (`/sound`, `/music_volume`, `/laughter`, `/laughter_volume`, `/laughter_offset`), so the patch's own controls show them and only work on their own while SMILE does not send (the patch loads with sound and laughter on, laughter level 0 dB); the speaker button in Max holds a click until the panel's `sound` changes
* laughter off: no new laugh starts and one that plays fades out
* **three speakers** on the PC's Realtek jacks: the two beside the camera (60° left and right of it) get the music in stereo and the laughter, the one opposite (180°) laughter only. Windows: Win+R → `mmsys.cpl` (the old Sound panel; also Settings → System → Sound → Advanced → More sound settings; the Windows 11 Settings app itself has no Configure) → Playback → Speakers (Realtek(R) Audio) (not Realtek Digital Output, the optical out) → Configure → **Quadraphonic** → Next → Finish (a stereo Windows configuration gives every driver 2 channels, and Max's `ad_mme` driver is stereo whatever Windows says); Properties → Advanced: note the format (24 bit, 48000 Hz), Enhancements: disable all; the Realtek Audio Console has the same choice (Speakers → Speaker Configuration) and gets no Speaker Fill or effects. Cables: green jack = front left and right (channels 1 and 2), black jack tip (its left) = back (channel 3)
* Max → Options → Audio Status: Driver **ad_portaudio Windows WASAPI**, Output Device **Speakers (Realtek(R) Audio)**, Sample Rate **48000** (the Windows format; Max's saved setting is 44100); I/O Mappings then lists 4 output channels. Max keeps this in its own settings, so it holds across restarts. If that driver lists 2 channels: `ad_asio` → `Realtek ASIO`, or `ad_portaudio Windows WDM-KS`. The patch plays the music on channels 1 and 2 (`ezdac~`) and the laughter on 1, 2 and 3 (`dac~ 1 2 3`: in Quadraphonic the black jack carries 3 and 4). A quick check for the back jack in Max: `cycle~ 440` → `*~ 0.1` → `dac~ 3`
* **where a laugh comes from**: the screen its grin starts on. `data\audio\screen_angles.txt` gives every screen its angle around the sphere (degrees seen from above, 0 = the camera, positive to the right), one line per screen as `index, angle;`: index = row × 6 + column of the player's grid (cell 0 top left; the Datapath unit PLn carries column n-1, its output k row k-1), + 24 for the player on 4116. The sphere has 12 ribs 30° apart, holding 5 and 3 screens in turn. The screens are numbered from the top of a 5-rib down, then the 3-rib to its right from the top down, and so on round the sphere (1-5, 6-8, 9-13, 14-16 …), and the Datapath outputs follow that numbering: screen s is unit ⌈s/4⌉ output ((s-1) mod 4)+1, player 1's units PL1-PL6 for screens 1-24, player 2's for 25-48. The file is written for that: screens 1-5 at 0°, 6-8 at 30°, up to 46-48 at -30° (player 1's ribs 0 … 150, player 2's 180 … -30). Only a screen's rib matters, not its height. On site: set the panel's `laughter_offset` to minus the angle of the rib the camera is in (it turns all angles at once); where the cabling differs from the numbering, edit the lines of those screens. The patch's `coll` reads it when the patch opens (plain text, no byte order mark; restart Max after a change). The speakers' own angles (-60, 60, 180) are in the three `expr`s of `p laughterSampler`, which pan each laugh over the two nearest speakers at constant power
* **laughter test** (Audio tab): `laughter_test` steps one laugh round the screens, `laughter_test_step` seconds each; the screen under test is white with its number meanwhile (1-48, the file's index + 1; `laughter_test_screen` shows it too). The laugh must come from the numbered screen, else that screen's line in `screen_angles.txt` is wrong. It is always the same laugh, the file `laughter_test_sample` of `data\audio\laughter` (SMILE sends the patch its number in the folder's alphabetical order, case ignored, which is the order `polybuffer~ readfolder` numbers the files in on NTFS; a laugh that is not the chosen file means that order differs), at one level, and the only one: SMILE sends Max `/laughter_test 1`, which closes the gate of the clips' laughs (the laughter switch may stay as it is, the test sounds either way), and bangs Max itself with `/test_bang <index> <sample>`, past the gates. New visitors wait while the test runs. The screens are lit by `/light <cell> <seconds> <number>` to the players on `player_1_port` and `player_2_port` (4117, 4118; `start_players.bat` starts them with `-l`, a player started without it takes no part)

### Lights (Art-Net)
* the Art-Net node is at the panel's `artnet.ip` (10.0.0.11, universe 1); `data\artnet.json` lists the modules: three cores (pc, audio, video) and the 24-LED ring around the camera (mako)

## RUN

### Start everything
* run ```scripts\startSMILE.bat```: it closes whatever runs of the installation (SMILE, the players, ComfyUI, Max) and starts it all again: ComfyUI in a minimized window, Max, the two players fullscreen on their monitors, SMILE borderless fullscreen on its own
* SMILE waits for ComfyUI and shows that on its screen: the camera image in pixels with a status line (`waiting for ComfyUI`, `warming up: ...`, sharp when ready). The status line also says `no camera frames` when the camera delivers none, and the generator's reason when ComfyUI stops answering
* the apps run under `pythonw`, without a console: what they say is in `output\logs\`, one file per start per app (`smile_`, `player1_`, `player2_`, `comfy_`), plus `crashes.log`, one line per unhandled exception across all starts. The first line of every SMILE and player log is the count: `No crashes recorded`, or `3 crashes since 2026-10-02 (output/logs/crashes.log), last ...`. Delete `crashes.log` to start the count over

### Settings panel
* open http://localhost:667 in a browser (also reachable from other devices on the network, the log says the address)
* presets live in `data\settings\`; `smile.json` loads at startup (the star in the panel picks the startup preset; it is kept in `data\settings\.ui_preset.json`)
* monitors are numbered primary first, then left to right: SMILE uses the one in its preset (`render.window.monitor`), the players the ones in `scripts\start_players.bat` (`PLAYER_1_MONITOR`, `PLAYER_2_MONITOR`); a player whose monitor is not connected opens in a window on the primary one

### Keys
* in the SMILE window: Enter starts a sequence, F fullscreen, W borderless fullscreen, Esc quits (the panel's Exit button does the same); fullscreen hides the mouse pointer unless the panel's window setting `fullscreen_mouse` is on (a window always shows it)
* in a player window: F borderless fullscreen, Esc quits

### Pieces on their own
* ```scripts\start_comfy.bat```: only ComfyUI, in this window (`--enable-manager` for the ComfyUI-Manager); `smile.py` does not start it, so start it first when running SMILE from VSCode
* ```.venv\Scripts\python smile.py```: `-fs` borderless fullscreen (else the preset's window mode), `-wc` a webcam instead of the Vimba camera, `-lm` landscape instead of portrait, `-v` DEBUG on the console
* ```scripts\start_players.bat```: (re)starts both players; `window` puts both in a window on the primary monitor, `console` gives them a console for the logs (Ctrl+C stops them), other words go to both players: ```start_players.bat console window -n 12```; ```.venv\Scripts\python player.py -h``` lists them
* ```scripts\make_test_workflow.bat```: writes `data\comfy\smile_test.json`, the app's workflow with a preset's values plus previews and an input switch, to drag into the ComfyUI page
* drop a saved clip or portrait on ```scripts\clip_info.bat``` to see what it was made with (the record: id, style, description, seed, settings); ```.venv\Scripts\python -m modules.save.record <file> --preset <name>``` writes those settings to `data\settings\<name>.json`
* tests: ```.venv\Scripts\python -m pytest``` from the repo root

### Credit
* the main screen always has a QR code to https://oostrik.net/smile in its bottom right corner, and shows a meme caption over the top of the portrait (in the comic font `data\Bangers-Regular.ttf`) with each visitor's grin and result: `Matthias Oostrik` and under it one of the lines of `data\credit_lines.txt` (one per line, `#` comments; edit it any time, each visitor draws a random one, never the last one again; `loves your smile!` when the file is empty or missing), the panel's Render tab has `credit` on or off, the name, the file, the line of the current visitor, the URL, the QR size, `credit_camera` (the caption also over the camera view while waiting, with a line of its own) and `credit_players` (the caption over every clip on the walls: the name over the top, a random line along the bottom, new at every play)
* each player wall has one cell with the QR code and the name under it, as wide as the code, instead of a clip, for the length of one clip, then on the next cell where a clip starts: `--credit_name`, `--credit_url`, `--no_credit` (```start_players.bat --no_credit```) leaves it out
* **scrolling name**: with the Render tab's `credit_scroll` on, the name runs along one ring of the sphere every `credit_scroll_every` seconds, over the clips, like a ticker, letters nearly a screen tall (`credit_scroll_padding` above and below): it enters at the right side of a random screen of one of the `credit_scroll_rows` (0 top to 4 bottom; 1 and 3 by default, the rings with 12 screens) and leaves at the left side of the last of `credit_scroll_screens_min` to `credit_scroll_screens_max` screens, at `credit_scroll_speed` screen widths per second. SMILE sends each pass to both players (`/scroll` on the test ports, so `start_players.bat -l`), which draw it by their own clock; `credit_scroll_now` sends one pass at once, `credit_scroll_last` shows the last one with its screens numbered as in the laughter test. On site: `credit_scroll_reverse` if the name runs the wrong way round the sphere, `credit_scroll_gap` the width of the wall between two screens (pixels that pass unseen, so a letter crosses it instead of jumping). The screens of a row follow the numbering of `screen_angles.txt`: a laugh from the numbered screen in the laughter test means the rows are right too

### Output
* `output\grins\A\` and `output\grins\B\` hold the grin clips (h264 mp4, the players read the style from the folder), `output\portraits\` the capture and the portrait of each visitor; `output\temp\` is where they are built. About 2.9 MB per visitor, nothing is cleaned up

## STARTUP
* make a shortcut to `scripts\startSMILE.bat`
* press Windows+R, type `shell:startup`, move the shortcut into that folder

## ON SITE CHECKS

A check passes when the installation does what is expected and the SMILE log of that run has no
`CRITICAL`, and no `ERROR` other than the ones named. The dates are when a check last passed on the
rig.

1. **Camera cable out, during the camera phase** (passed 2026-10-02): `no camera frames` on the image after about 3 s, no flash, the panel's `device` goes to `none`; log: `Vimba camera: no complete frame for 5 s, opening the camera again`. Cable back in: the image returns within about 10 s, log: `Vimba camera delivers frames again`
2. **Start with the camera unplugged, then plug it in** (passed 2026-10-02): `no camera frames` from the start, `Vimba camera: no camera found` once in the log; after plugging in, the image appears by itself within about 10 s
3. **Monitor cable out while SMILE runs**, once fullscreen, once in a window (passed 2026-10-02): SMILE keeps running (log: at most `Error in event callback`), and the image is back on its own monitor when the cable is in again
4. **ComfyUI stopped during a visitor** (close its window) (open): the visitor's own capture stands as the result, then the camera in pixels with `waiting for ComfyUI at 127.0.0.1:8188`, nothing of the previous visitor; ComfyUI started again: warm-up, then ready, without restarting SMILE
5. **Reset during a warm-up** (open): the status stays `warming up: ...` and ends in ready
6. **An evening unattended** (open): afterwards no `WARNING`, `ERROR` or `CRITICAL` in the log that is not explained, `dropped_frames` low in the panel, `last_duration` well under `job_timeout` (45 s)
7. **Light ring at the end of a visitor** (open): stays full red into the bloom, no dip
8. **Music** (open): two clips that start at the same moment send two `/bang`s, each with its cell; Max plays two laughs, each from its own screen
9. **A failed visitor** (open): a manual start (Enter) with nobody in the image fails in ComfyUI: flash, the camera frame as the result, bloom, back to the camera
10. **Both player monitors connected** (open): player 2 fullscreen on its own screen (on the dev PC it runs in a window: `Monitor 2 is not connected`)
11. **Packet size** (open): the log line at start says `packet size 8228` or more; with 1500, watch `dropped_frames`
12. **Startup after a reboot** (open): everything comes up by itself through the shortcut, SMILE waits for ComfyUI with its status line
13. **Network stays up for a day** (open): in the System log the adapter driver (`aqnic650`) reports `Network link is lost` only at boot, and `Microsoft-Windows-NCSI/Operational` has no `ArpProbeFailed`; the SMILE log has no `no complete frame for 5 s`
14. **Three speakers** (open): after a restart Max's Audio Status still shows the WASAPI driver with 4 output channels and no error in the Max console (`coll` found its file); the music is stereo on the front pair only; with the Audio tab's `laughter_test` on, the white screen with its number walks round the sphere, 1 to 48, the same laugh each time and no other (a clip starting meanwhile stays silent), and every laugh comes from the numbered screen (one near the back speaker from the back, one beside the camera from that side); turning the panel's `laughter_offset` by 180 swaps them; the test off: the readout shows 0, the clips' laughs are back and the next visitor starts
15. **Credit** (open): a phone scans the QR code from the main screen and from a player wall's credit cell, at the distance visitors stand and from a phone video of the screen; both open https://oostrik.net/smile
16. **Scrolling name** (open): `credit_scroll_now` with both players running: the name appears on row 1 or 3, crosses from one player to the other without a jump and leaves at a screen's side; it reads the right way round (else `credit_scroll_reverse`); `credit_scroll_gap` set so a letter crossing the wall between two screens looks continuous; the letter height and `credit_scroll_every` chosen; the pass is in no recording and does not disturb the laughter test

## KNOWN LIMITS
* no automatic restart: an app that ends by itself stays down until `startSMILE.bat` is run; `crashes.log` says what happened
* an app that hangs without ending is not detected either
* no messages to a phone when something fails
* `output\` grows about 2.9 MB per visitor; nothing is cleaned up
* on a camera path that does not pass jumbo frames the packet size stays 1500 (see the camera section)
* `data\audio\screen_angles.txt` assumes the Datapath outputs are cabled in the screens' numbering (unit PL1 output 1 = screen 1, the top of the first 5-rib, going down and then to the right); check it on site (a laugh from a known screen, `laughter_offset` for the camera's rib). A screen without an entry leaves its laugh where the previous one was

## HINTS
* Could not find the DLL(s) 'msvcp140.dll: https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170
