/* SMILE Online, the phone page. One portrait frame under the title of the machine, the artist's link under
   it, and behind everything the same picture as a field of smileys (THE HALFTONE). The frame is the camera:
   in coarse pixels while you wait, the block from your place in line; when it is your turn, your live video,
   in pixels as long as the face is not right, the worse the coarser, sharp when it is (the face data comes
   from SMILE, the drawing happens here, so only the judging lags; nothing is drawn on the image), and the
   charge is the ring light: the page goes white behind the frame as it fills, quickly at first, full white at
   the flash. From the flash the window is the main screen of the installation as it is rendered, frame by
   frame, to the end of its sequence, while the white fades with the installation's flash, and the visitor's
   track plays from the flash as in the room: on the tape, spun up from standstill at the flash, muffled and
   opening up through the transformation, from the room's own stage progress (the Max patch's curves), within
   what a phone speaker can do: the filter never closes below PHONE_FLOOR and the tape never runs under 0.3
   (it is silent instead), as the patch has it. The track is named at the turn's start and fetched and decoded
   during the judging; it plays through Web Audio, a media element would be silenced at the tape's low speeds.
   The turn is over after the fade, but the track plays on until the saved video is there (fetched while the
   room still shows the result, so it mostly is): then it winds down to a stop and the video takes over,
   looping with its own sound. The room's laughter samples play at random from the moment of consent, panned
   about, while nothing else sounds.
   Under the frame stands the bar, the page's one button and one line of text (THE BAR): I AGREE TO SMILE
   under the terms, the wait (WAITING FOR SMILE, N BEFORE YOU, SMILE IS ASLEEP), in the turn what the face
   still needs, TURN YOUR VOLUME UP once the flash's white is full, BULLSHITTING over the loading bar of the
   generation and the grin (poops fly out of it), PREPARING YOUR SMILE over the timeout, SHARE YOUR SMILE once
   the turn is over and the file is on the phone; LOST YOU, MISSED YOU, NO SMILE THIS TIME (a turn that left no
   video) and NO CAMERA, NO SMILE when so. With
   ?debug in the address the page shows its state in a corner (THE READOUT), the only other text.
   The camera is drawn as a mirror and SENT as a mirror too: the installation and the portrait then show
   what you saw on the phone, not its mirror image.
   Messages: see relay/server.py. The order here: what the page is made of, the state of the turn, the bar,
   the sound, the relay's messages, the camera and the drawing, the sharing, the boot at the end. */

'use strict';

// THE PAGE: what the script finds in index.html, and the canvases it draws on
const FRAME = 0;        // a binary message: our camera frame to SMILE
const SHOW = 1;         // a binary message: a frame of the main screen from SMILE
const LAST = 2;         // the turn's last frame, without the caption: the window fades to it and keeps it
const CAPTION_FADE = 0.5;   // s, the caption's fade on the main screen (CreditLayer.FADE_S)
const W = 640, H = 800; // the frame we send and draw, 4:5 like the installation's image
const FPS = 15;         // frames a second to SMILE during the turn
const QUALITY = 0.6;    // JPEG
const MAX_BLOCK = 32;   // a face's pixel size at distance 1 (the installation's face_effect)
const TAGS = '#smile #oostrik';   // the share text, until the real tags are decided
const CELL = 20;        // css pixels per smiley of the background
const NBSP = '\u00a0';             // the empty label's one character: the bar keeps its height
const POOP = '\uD83D\uDCA9';       // the poop emoji, as the phone draws it
const ACCENTS = /[\u0300-\u036f]/g;  // the accents normalize('NFD') splits off, dropped from the file name

const $ = (id) => document.getElementById(id);
const cam = $('cam'), view = $('view'), grinVideo = $('grin'), light = $('light'), flash = $('flash');
const share = $('share'), halftone = $('halftone'), debugLine = $('debug');
const frame = document.createElement('canvas');     // the camera as we send it: mirrored
const small = document.createElement('canvas');     // for the pixelation
view.width = frame.width = W;
view.height = frame.height = H;
const viewCtx = view.getContext('2d'), frameCtx = frame.getContext('2d'), smallCtx = small.getContext('2d');
const before = document.createElement('canvas');    // the frame shown when the last one comes, for the fade between them
before.width = W;
before.height = H;
const beforeCtx = before.getContext('2d');

// THE STATE of the turn, shared by the sections below
let ws = null;
let state = 'idle';     // idle, waiting, you, transform, done, asleep, lost, nocam (the body's class too)
let eye = {};           // the last state message from the relay
let faces = null;       // the last face message during the turn
let stream = null;      // the camera
let result = null;      // the result message: urls and caption, after the turn
let shareFile = null;   // the video as a File, fetched when the result comes, so the share sheet opens at once
let fetched = null;     // the fetch of that file, awaited by the video and the share sheet
let flashAt = 0;        // performance.now() of the flash: the track starts there
let streaming = false;  // the window shows the main screen: from full white on, the flash-in is the phone's own image going white
let ended = false;      // the last frame came: the picture stands, frames after it are not drawn
let lastStage = '';     // the last stage message, for the readout
let expired = false;    // the relay said the pass of this page ran out: no reconnecting, the visitor scans again
const timing = {};      // moments of the turn's end (ms since the page loaded): what took how long, for the relay's log

function setState(next) {
  state = next;
  document.body.className = next;
  if (next === 'done') document.body.classList.add('colour');
  if (next !== 'you') faces = null;
  if (next !== 'you' && next !== 'transform') setLight(0, 0.25);
  if (next !== 'done') { grinVideo.pause(); document.body.classList.remove('playing'); }
}

function subjectFace(message) {
  // the face the judging is about: the one SMILE flags (the largest qualifying, else the largest), else the
  // widest of the message, else null
  const list = message.faces || [];
  const width = (face) => (face.box ? face.box[2] - face.box[0] : 0);
  return list.find((face) => face.subject) || list.slice().sort((a, b) => width(b) - width(a))[0] || null;
}

function resultUrl(message) {
  return message.grin || message.portrait;     // the video, or the portrait when there is no video
}

// THE LIGHT AND THE FLASH. The ring light: the white over the halftone, behind the frame, 0..1, reached in
// `seconds`; the flash: the white over the window, the room's flash as its progress says
function setLight(value, seconds) {
  light.style.transitionDuration = `${seconds}s`;
  document.documentElement.style.setProperty('--light', String(value));
}

function lightNow() {
  return parseFloat(document.documentElement.style.getPropertyValue('--light')) || 0;
}

function setFlash(value, seconds) {
  flash.style.transitionDuration = `${seconds}s`;
  flash.style.opacity = String(value);
}

// THE BAR under the frame, the page's one button and one line of text. One block: its texts, its fill and
// everything that moves in it live here, and the rest of the page only tells it the turn's moments (bar.turn,
// bar.flash, bar.stage, bar.done ...). It is a box, not a button element: a button clips its contents, and
// the loud word grows past the bar. The label stands invisible in the flow for its room and the bar's height,
// and two copies over it are the text seen: the white one clipped to the black part of the bar, the black one
// to the fill (style.css).
const bar = (() => {
  // the texts: "(ai)" is the machine's mark, the ring with "ai" in it; *word* is the loud word
  const AGREE = 'I AGREE TO SMILE(ai)', SHARE = 'SHARE YOUR SMILE(ai)', BULLSHITTING = 'BULLSHITTING(ai)';
  const VOLUME = 'TURN YOUR *VOLUME* UP', PREPARING = 'PREPARING YOUR SMILE(ai)', FAILED = 'NO SMILE(ai) THIS TIME';
  // what the face still needs, from the judging's reject label (the panel's words, the measure first:
  // `yaw 32 > 20`, see modules/face/quality.py rejection_label); HOLD IT when it is right
  const HINTS = { conf: 'IS THAT A FACE?', eyes: 'COME CLOSER', edge: 'INTO THE MIDDLE', yaw: 'LOOK STRAIGHT AT ME',
                  roll: 'HEAD STRAIGHT', open: 'OPEN YOUR EYES' };
  // the labels that move get a letter each: I AGREE TO SMILE and SHARE YOUR SMILE, the calls to action, ripple
  // (a pop running through the words, 60 ms a letter, then rest); BULLSHITTING stands still, its letters are
  // elements too (no motion of their own) so one jolts when poops leave it; the other texts stand still
  const MOTION = { [BULLSHITTING]: ['letter', 0], [AGREE]: ['ripple', 0.06], [SHARE]: ['ripple', 0.06] };
  const VOLUME_HOLD = 5000;       // ms the volume is asked for at least, before BULLSHITTING takes the bar
  // the loud word's cycle and the point in it from which the word stands at its size (the keyframes agree:
  // 2.4 s, still from 55 %); a label change waits for that, so the bar does not switch mid-bounce
  const LOUD_CYCLE = 2400, LOUD_STILL = 0.55 * LOUD_CYCLE;
  // the loading bar: the diffusion and the grin are one fill: the diffusion fills it to DIFF_END; when its
  // progress has then stood still for CREEP_AFTER (the portrait waits for the grin: the room's player holds it
  // until the frames come in fast enough to play without a stall, a wait the page cannot know, and the room
  // sends no stage meanwhile), the bar creeps on from where it is toward CREEP_TO, slower and slower, on the
  // bar's own clock (the tick), so it never stands still and never arrives; the grin takes it from wherever it
  // is to full. The timeout is a fill of its own; full from the bloom on
  const DIFF_END = 0.1, CREEP_TO = 0.25, CREEP_AFTER = 1000, CREEP_TAU = 6000;    // the creep: a dead time and a time constant, ms
  const NOTICE_HOLD = 3000;       // ms MISSED YOU and LOST YOU stay before the wait may take the bar
  // the poops: while the bar says BULLSHITTING they fly out of the word, three or four at a time from a random
  // letter every while, each thrown its own way (any direction, its own speed, spinning, falling under
  // gravity, fading, CSS: fly) and gone when it has flown; they stand outside the label, so they keep their
  // colour over the fill
  const POOP_EVERY = 150, POOPS_IN_AIR = 60;
  const POOP_SPEED = [60, 150], POOP_GRAVITY = 100, POOP_FLIGHT = [1300, 2200];     // px/s, px/s^2, ms

  const label = $('label');
  let running = false;            // from the flash to the turn's end: the stages count, the tick runs
  let progress = 0;               // the fill, 0..1
  let bullshitting = false, volumeAsked = false, volumeAskedAt = 0, preparing = false;
  let grinFrom = null;            // the fill when the grin began
  const stageNow = { stage: null, p: 0 };          // the last stage
  const creep = { p: null, since: 0, from: 0 };   // the diffusion's progress, since when it stands, the fill then
  let tick = null, poopTimer = null;
  let noticeUntil = 0;            // until when a notice (MISSED YOU, LOST YOU) holds the bar

  // THE LABEL
  function mark() {
    const ring = document.createElement('span');
    ring.className = 'ai';
    ring.textContent = 'ai';
    return ring;
  }
  function loud(text) {
    // the text with its *loud words* as elements: a fragment of text nodes and <i class="loud">
    const piece = document.createDocumentFragment();
    text.split(/\*([^*]+)\*/).forEach((part, n) => {
      if (n % 2 === 0) { if (part) piece.appendChild(document.createTextNode(part)); return; }
      const word = document.createElement('i');
      word.className = 'loud';
      word.textContent = part;
      piece.appendChild(word);
    });
    return piece;
  }
  function lettersOf(text) {
    // the letter elements of a moving label (not the mark's)
    return Array.from(text.querySelectorAll('i')).filter((i) => !i.querySelector('.ai'));
  }
  function say(text, still = false) {
    // the label; `still`: the text without its motion (the agree button before the camera is allowed)
    const motion = still ? null : MOTION[text];
    stopPoop();
    const pieces = (text || NBSP).split('(ai)');      // the text between the marks; empty: the bar keeps its height
    label.textContent = '';
    let i = 0;
    pieces.forEach((piece, n) => {
      const letters = motion ? Array.from(piece) : [piece];
      if (n < pieces.length - 1) letters.push(null);        // the mark after this piece
      for (const letter of letters) {
        const node = letter === null ? mark() : document.createTextNode(letter);
        if (!motion) { label.appendChild(letter === null ? node : loud(letter)); continue; }
        const step = document.createElement('i');
        step.className = motion[0];
        step.style.animationDelay = `${i++ * motion[1]}s`;
        step.appendChild(node);
        label.appendChild(step);
      }
    });
    for (const copy of share.querySelectorAll('.copy .text')) copy.innerHTML = label.innerHTML;    // the text as seen
    if (text === BULLSHITTING) flyPoops();
  }
  function whenLoudStill(fn) {
    const word = share.querySelector('i.loud');
    try {
      const motion = word && word.getAnimations()[0];
      if (motion && motion.currentTime !== null) {
        const t = motion.currentTime % LOUD_CYCLE;
        if (t < LOUD_STILL) { setTimeout(fn, LOUD_STILL - t); return; }
      }
    } catch (error) { /* an old browser: now */ }
    fn();
  }

  // THE FILL AND THE TAP
  function fill(value) {
    if (value < progress) {
      // a new stage or a new turn: the fill jumps back to the start instead of running back
      share.classList.add('snap');
      requestAnimationFrame(() => requestAnimationFrame(() => share.classList.remove('snap')));
    }
    progress = value;
    share.style.setProperty('--progress', String(value));
  }
  function tappable(on) {
    // the bar takes a tap as the agree button and as the share button, and none in between
    share.classList.toggle('off', !on);
    share.setAttribute('aria-disabled', on ? 'false' : 'true');
  }
  function stageProgress(stage, p) {
    // the fill for a stage's progress (null: not a stage of the bar)
    if (stage === 'diffusion') {
      const now = performance.now();
      const base = DIFF_END * Math.min(1, p);
      if (p !== creep.p) { creep.p = p; creep.since = now; creep.from = Math.max(base, progress); }
      const still = now - creep.since - CREEP_AFTER;      // how long the progress has stood, past the dead time
      return Math.max(base, creep.from + (still > 0 ? (CREEP_TO - creep.from) * (1 - Math.exp(-still / CREEP_TAU)) : 0));
    }
    if (stage === 'grin') {
      if (grinFrom === null) grinFrom = progress;
      return grinFrom + (1 - grinFrom) * Math.min(1, p);
    }
    if (stage === 'timeout') return Math.min(1, p);
    if (stage === 'bloom' || stage === 'fade') return 1;
    return null;
  }
  function startTick() {
    stopTick();
    tick = setInterval(() => { if (stageNow.stage === 'diffusion') fill(stageProgress('diffusion', stageNow.p)); }, 100);
  }
  function stopTick() {
    clearInterval(tick);
    tick = null;
  }

  // THE POOPS
  function flyPoops() {
    const letters = lettersOf(label);
    if (!letters.length) return;
    const one = () => {
      if (share.querySelectorAll('i.fly').length >= POOPS_IN_AIR) return;     // enough in the air
      const at = Math.floor(Math.random() * letters.length);
      const from = letters[at];
      for (const text of [label, ...share.querySelectorAll('.copy .text')]) {
        const letter = lettersOf(text)[at];      // the same letter in the label and in both copies
        letter.classList.remove('jolt');
        void letter.offsetWidth;      // the jolt starts over
        letter.classList.add('jolt');
      }
      const count = 3 + Math.floor(Math.random() * 2);
      for (let n = 0; n < count; n++) {
        const poop = document.createElement('i');
        poop.className = 'fly';
        poop.textContent = POOP;
        poop.style.left = `${label.offsetLeft + from.offsetLeft}px`;
        poop.style.top = `${label.offsetTop + from.offsetTop}px`;
        const angle = 2 * Math.PI * Math.random();      // thrown out of the word, any way
        const speed = POOP_SPEED[0] + (POOP_SPEED[1] - POOP_SPEED[0]) * Math.random();
        const flight = Math.round(POOP_FLIGHT[0] + (POOP_FLIGHT[1] - POOP_FLIGHT[0]) * Math.random());
        const seconds = flight / 1000;
        poop.style.setProperty('--vx', `${(Math.cos(angle) * speed * seconds).toFixed(0)}px`);
        poop.style.setProperty('--vy', `${(Math.sin(angle) * speed * seconds).toFixed(0)}px`);
        poop.style.setProperty('--drop', `${(0.5 * POOP_GRAVITY * seconds * seconds).toFixed(0)}px`);
        poop.style.setProperty('--spin', `${((Math.random() - 0.5) * 540).toFixed(0)}deg`);
        poop.style.animationDuration = `${flight}ms`;
        share.appendChild(poop);
        setTimeout(() => poop.remove(), flight + 100);
      }
    };
    one();
    poopTimer = setInterval(one, POOP_EVERY);
  }
  function stopPoop() {
    clearInterval(poopTimer);
    poopTimer = null;
    for (const poop of share.querySelectorAll('i.fly')) poop.remove();
  }

  // THE TEXTS OF THE TURN
  function lineText(message) {
    // the wait, from the relay's state (its own sentence, `status`, is for the API): asleep when SMILE is not
    // there, else the place in line
    if (message.eye === 'asleep') return 'SMILE(ai) IS ASLEEP';
    const ahead = Math.max(0, (message.place || 1) - 1);
    return ahead ? `${ahead} BEFORE YOU` : 'WAITING FOR SMILE(ai)';
  }
  function faceHint(subject) {
    if (!subject) return 'SHOW ME YOUR FACE';
    if (!subject.reject) return 'HOLD IT';
    const [what, value] = subject.reject.split(' ');
    if (what === 'pitch') return parseFloat(value) < 0 ? 'CHIN UP' : 'CHIN DOWN';    // the pitch is positive looking up
    return HINTS[what] || 'SHOW ME YOUR FACE';
  }
  function tell(text) {
    // a text outside a turn: black, empty, no tap
    running = false;
    stopTick();
    tappable(false);
    fill(0);
    say(text);
  }
  function notice(text) {
    // a text that holds against the wait's for a while (the relay sends the state right after a turn's end)
    tell(text);
    noticeUntil = performance.now() + NOTICE_HOLD;
  }
  function offer(ready) {
    // the bar is the button once the video is here on the phone; until then it is the loading bar
    tappable(ready);
    say(ready ? SHARE : BULLSHITTING);
  }

  return {
    fill,
    tappable,
    agree(moving) {       // under the terms: full white, the one tap; still until the camera is allowed
      fill(1);
      tappable(true);
      say(AGREE, !moving);
    },
    wait(message) { if (performance.now() >= noticeUntil) tell(lineText(message)); },
    lost() { notice('lost you'); },
    missed() { notice('missed you'); },
    expired() { tell('link expired, scan SMILE(ai) again'); },
    nocam() { tell('no camera, no SMILE(ai)'); },
    turn() {              // a new turn: everything of the previous one goes
      bullshitting = false; volumeAsked = false; volumeAskedAt = 0; preparing = false;
      grinFrom = null; creep.p = null; stageNow.stage = null; stageNow.p = 0;
      tell('SHOW ME YOUR FACE');
    },
    hint(message) { say(faceHint(subjectFace(message))); },
    flash() {             // the picture is taken: empty; TURN YOUR VOLUME UP once the white is full (volume)
      running = true;
      tappable(false);
      fill(0);
      say('');
      startTick();
    },
    volume() {            // the white is full, the track starts: the one moment to ask for the volume, once
      if (volumeAsked) return;
      volumeAsked = true;
      volumeAskedAt = performance.now();
      say(VOLUME);
    },
    stage(name, p) {      // a stage of the sequence with its progress: the fill, and the texts along the way
      if (name === 'timeout' && !preparing) {
        // the grin is done, the file is on its way: the bar prepares, still, without the poops
        preparing = true;
        say(PREPARING);
      }
      if ((name === 'diffusion' || name === 'grin') && !bullshitting) {
        // the generation: BULLSHITTING from here, but not before TURN YOUR VOLUME UP has had its time (the
        // generation can start within a second of the flash out), and once VOLUME is back at its size
        bullshitting = true;
        const wait = Math.max(0, volumeAskedAt + VOLUME_HOLD - performance.now());
        setTimeout(() => whenLoudStill(() => { if (running && bullshitting && !preparing) offer(false); }), wait);
      }
      stageNow.stage = name;
      stageNow.p = p;
      const value = stageProgress(name, p);
      if (value !== null) fill(value);
    },
    done(ready) {         // the turn is over: full; the share button when the file is here, else BULLSHITTING until it is
      running = false;
      stopTick();
      fill(1);
      offer(ready);
    },
    failed() {            // the turn is over and left no video: nothing to play or share
      running = false;
      stopTick();
      tappable(false);
      fill(1);
      say(FAILED);
    },
    offer,                // the file came after the turn's end
  };
})();

// THE SOUND: browsers only play sound after a tap, so the "I agree" tap wakes the audio context (the
// track and the laughter play through it) and primes the video element with a moment of silence, so it
// may play with sound later without another tap.
let audioCtx = null, musicGain = null, musicFilter = null;    // the track's chain: source -> low-pass filter -> volume

function silentWav() {
  const samples = 441, header = 44, bytes = new Uint8Array(header + samples * 2), data = new DataView(bytes.buffer);
  const text = (at, s) => { for (let i = 0; i < s.length; i++) bytes[at + i] = s.charCodeAt(i); };
  text(0, 'RIFF'); data.setUint32(4, 36 + samples * 2, true); text(8, 'WAVE'); text(12, 'fmt ');
  data.setUint32(16, 16, true); data.setUint16(20, 1, true); data.setUint16(22, 1, true);
  data.setUint32(24, 44100, true); data.setUint32(28, 88200, true); data.setUint16(32, 2, true); data.setUint16(34, 16, true);
  text(36, 'data'); data.setUint32(40, samples * 2, true);
  return URL.createObjectURL(new Blob([bytes], { type: 'audio/wav' }));
}

function unlockAudio() {
  try {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    musicFilter = audioCtx.createBiquadFilter();
    musicFilter.type = 'lowpass';
    musicFilter.frequency.value = 20000;
    musicFilter.Q.value = 1;
    musicGain = audioCtx.createGain();
    musicFilter.connect(musicGain).connect(audioCtx.destination);
    audioCtx.resume();
  } catch (error) { audioCtx = null; musicGain = null; musicFilter = null; }
  grinVideo.muted = false;
  grinVideo.src = silentWav();
  grinVideo.play().then(() => { grinVideo.pause(); grinVideo.removeAttribute('src'); grinVideo.load(); }).catch(() => {});
  loadLibrary();
  scheduleLaugh();
  flushPendingTrack();
}

async function fetchAudio(url) {
  // a sound file, decoded for the audio context
  const response = await fetch(url);
  if (!response.ok) throw new Error(`fetch ${response.status}`);
  return audioCtx.decodeAudioData(await response.arrayBuffer());
}

// THE TRACK, through Web Audio: fetched and decoded when named, started at the flash (or as soon as it is
// decoded, that far in), played on the tape, wound down when the video is there
let track = { id: '', buffer: null, source: null, playing: false, error: '' };    // this turn's track, decoded, and its player
let pendingTrack = null;    // a track named before the sound was unlocked: fetched at the unlock

function flushPendingTrack() {
  if (pendingTrack) { const { id, url } = pendingTrack; prepareTrack(id, url); }
}

async function prepareTrack(id, url) {
  if (track.id === id) return;
  if (!audioCtx) { pendingTrack = { id, url }; return; }     // fetched at the unlock
  pendingTrack = null;
  stopTrack(0);
  track = { id, buffer: null, source: null, playing: false, error: '' };
  try {
    const buffer = await fetchAudio(url);
    if (track.id !== id) return;
    track.buffer = buffer;
    if (state === 'transform' && !track.playing) startTrack((performance.now() - flashAt) / 1000);
  } catch (error) {
    if (track.id === id) track.error = String(error && error.message || error);   // the turn goes on without one; ?debug shows it
  }
}

function startTrack(offset) {
  if (!audioCtx || !track.buffer || track.playing) return;
  audioCtx.resume();
  const source = audioCtx.createBufferSource();
  source.buffer = track.buffer;
  source.loop = true;            // as the patch's groove~: a long wait for the video does not end it
  source.playbackRate.value = rateFor(tape.target);
  source.connect(musicFilter);
  const t = audioCtx.currentTime;
  musicGain.gain.cancelScheduledValues(t);
  musicGain.gain.setValueAtTime(gateFor(tape.target), t);
  source.start(t, offset % track.buffer.duration);
  track.source = source;
  track.playing = true;
}

function takeSource() {
  // the player, taken out of the track (it plays on until it is stopped)
  const source = track.source;
  track.source = null;
  track.playing = false;
  return source;
}

function fadeGainOut(hold, end) {
  // the volume kept where it is until `hold`, then down to 0 by `end` (audio context times)
  musicGain.gain.cancelScheduledValues(audioCtx.currentTime);
  musicGain.gain.setValueAtTime(musicGain.gain.value, hold);
  musicGain.gain.linearRampToValueAtTime(0, end);
}

function stopTrack(seconds) {
  // the track faded out over `seconds` (0: at once); the laughter may sound again
  const source = takeSource();
  laughMuted = false;
  if (!source || !audioCtx) return;
  const t = audioCtx.currentTime;
  try {
    if (seconds > 0) {
      fadeGainOut(t, t + seconds);
      source.stop(t + seconds);
    } else {
      source.stop(t);
    }
  } catch (error) { /* already stopped */ }
}

function windDown(seconds, done) {
  // the wind-down: the speed from where it is to a stop over `seconds` (the patch's fade), the volume held
  // until the last fifth, then `done`
  const source = track.source;
  if (!source || !audioCtx) { stopTrack(0); done(); return; }
  const t = audioCtx.currentTime;
  tape.target = 0;
  source.playbackRate.cancelScheduledValues(t);
  source.playbackRate.setValueAtTime(source.playbackRate.value, t);
  source.playbackRate.linearRampToValueAtTime(0, t + seconds);
  fadeGainOut(t + seconds * 0.8, t + seconds);
  takeSource();
  setTimeout(() => { try { source.stop(); } catch (error) { /* stopped */ } laughMuted = false; done(); }, seconds * 1000);
}

// THE TAPE: the speed and the filter of the track by the room's stage and progress, the Max patch's
// curves (p speedVar, p musicMixer in Smile-Music-Player.maxpat; modules/audio/tape.py has the same spin-up)
const SPIN_UP = [[0, 0], [0.122, 0.333], [0.559, 0.573], [0.707, 1.0], [0.814, 0.653], [1.0, 0.773]];   // x progress, y speed / 1.5
const FLASH_IN_FREQ = [[0, 0.57], [0.139, 0.725], [1, 0.0078]];      // x progress^0.5, y of the 150..20000 Hz range
const FLASH_OUT_FREQ = [[0, 0], [0.0265, 0], [1, 0.418]];
const DIFFUSION_FREQ = [[0, 0.463], [0.152, 0.254], [0.748, 1], [1, 1]];   // x progress^3.33
const PHONE_FLOOR = 1200;       // Hz: a phone speaker has nothing under this, the room's 150 Hz would be silence
const PHONE_Q = 2.5;            // the patch's resonance of 11 is a boom the speaker cannot make either
const STANDSTILL = 0.3;         // the patch's `< 0.3`: under this the tape stands still (the source is silent, not slowed to 0)
const tape = { target: 1, freq: 20000, q: 1 };

const rateFor = (speed) => Math.max(STANDSTILL, speed);
const gateFor = (speed) => (speed < STANDSTILL ? 0 : 1);

function curve(points, x) {
  if (x <= points[0][0]) return points[0][1];
  for (let i = 1; i < points.length; i++) {
    if (x <= points[i][0]) {
      const [x0, y0] = points[i - 1], [x1, y1] = points[i];
      return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
    }
  }
  return points[points.length - 1][1];
}

function tapeReset() {
  tape.target = 0; tape.freq = PHONE_FLOOR; tape.q = PHONE_Q;
  applyTape(0);
}

function tapeStage(stage, p) {
  const hz = (v) => PHONE_FLOOR + v * (20000 - PHONE_FLOOR);
  switch (stage) {
    case 'flash_in': {
      const s = 1.5 * curve(SPIN_UP, p);
      tape.target = s < STANDSTILL ? 0 : s;
      const r = Math.sqrt(p);
      tape.freq = hz(curve(FLASH_IN_FREQ, r)); tape.q = PHONE_Q + (0.85 - PHONE_Q) * r;
      break;
    }
    case 'flash_out':
      tape.target = 1; tape.freq = hz(curve(FLASH_OUT_FREQ, p)); tape.q = 1;
      break;
    case 'diffusion':
      tape.target = 1; tape.freq = hz(curve(DIFFUSION_FREQ, Math.pow(p, 3.33))); tape.q = 1;
      break;
    case 'grin': case 'timeout': case 'bloom': case 'fade':
      // open all the way; the room winds down in the bloom and the fade, the phone's track plays on until
      // the video is there (finishWithVideo)
      tape.target = 1; tape.freq = 20000; tape.q = 1;
      break;
    default:
      return;
  }
  applyTape(0.1);
}

function applyTape(seconds) {
  // the speed and the filter glide to their targets in `seconds` (the patch's `line` of a few ms)
  if (!audioCtx) return;
  const t = audioCtx.currentTime;
  if (musicFilter) {
    musicFilter.frequency.setTargetAtTime(tape.freq, t, seconds / 2 || 0.001);
    musicFilter.Q.setTargetAtTime(tape.q, t, seconds / 2 || 0.001);
  }
  if (track.source) {
    const rate = track.source.playbackRate;
    rate.cancelScheduledValues(t);
    rate.setValueAtTime(rate.value, t);
    rate.linearRampToValueAtTime(rateFor(tape.target), t + Math.max(0.01, seconds));
    musicGain.gain.setTargetAtTime(gateFor(tape.target), t, 0.02);
  }
}

// THE LAUGHTER: the room's samples (the relay's library), one every second or two at a random place while
// there is no music (the wall laughs all the time); none over the track or the video
let laughs = [];            // the laughter samples' urls, from the relay's library
const laughBuffers = new Map();
let laughMuted = false;     // from the flash until the track has stopped: the laughter is for the waiting, not over the music
let laughTimer = null;

async function loadLibrary() {
  try {
    const response = await fetch('/api/library');
    const library = await response.json();
    laughs = (library.laughs || []).map((name) => `/files/laughs/${name}`);
  } catch (error) { laughs = []; }
}

function scheduleLaugh() {
  if (laughTimer) clearTimeout(laughTimer);
  laughTimer = setTimeout(() => { laugh(); scheduleLaugh(); }, 800 + Math.random() * 1700);
}

async function laugh() {
  if (!audioCtx || !laughs.length || laughMuted || track.playing || state === 'transform' || !grinVideo.paused || eye.laughter === false) return;
  const url = laughs[Math.floor(Math.random() * laughs.length)];
  try {
    let buffer = laughBuffers.get(url);
    if (!buffer) {
      buffer = await fetchAudio(url);
      if (laughBuffers.size >= 40) laughBuffers.delete(laughBuffers.keys().next().value);
      laughBuffers.set(url, buffer);
    }
    const source = audioCtx.createBufferSource();
    source.buffer = buffer;
    const pan = audioCtx.createStereoPanner ? audioCtx.createStereoPanner() : null;
    const gain = audioCtx.createGain();
    gain.gain.value = 0.8 * Math.pow(10, (Math.random() * 6 - 3) / 20);
    if (pan) { pan.pan.value = Math.random() * 2 - 1; source.connect(pan).connect(gain); } else source.connect(gain);
    gain.connect(audioCtx.destination);
    source.start();
  } catch (error) { /* a sample that does not play is skipped */ }
}

// THE RELAY: the websocket, its messages (relay/server.py has the protocol) and the turn they drive
function connect() {
  const url = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/phone';
  ws = new WebSocket(url);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => {
    if (stream && state !== 'idle') join();
  };
  ws.onmessage = (event) => {
    if (typeof event.data === 'string') onMessage(JSON.parse(event.data));
    else onImage(event.data);
  };
  ws.onclose = () => {
    if (expired) return;
    if (state === 'you' || state === 'transform' || state === 'waiting') lostYou();
    setTimeout(connect, 2000);
  };
  ws.onerror = () => ws.close();
}

function send(message) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(message));
}

function join() {
  send({ type: 'join' });
}

function lostYou() {
  // the relay or the turn went: back in the line when the connection is back
  setState('lost');
  bar.lost();
}

function onMessage(message) {
  switch (message.type) {
    case 'state':
      eye = message;      // under the terms the bar is the agree button; the wait shows after the tap
      if (state === 'waiting' || state === 'asleep' || state === 'lost') {
        setState(message.eye === 'asleep' ? 'asleep' : 'waiting');
        bar.wait(message);
      }
      break;
    case 'turn':
      startTurn();
      break;
    case 'music':
      // this turn's track, named at the turn's start: fetched and decoded now, started at the flash (or that far in)
      if (state === 'you' || state === 'transform') prepareTrack(message.id, message.url);
      break;
    case 'face':
      if (state === 'you') {
        faces = message;
        const charge = message.charge || 0;
        setLight(1 - (1 - charge) * (1 - charge), 0.25);     // white quickly at first, then the last stretch
        bar.hint(message);
      }
      break;
    case 'stage':
      onStage(message);
      break;
    case 'result':
      onResult(message);
      break;
    case 'turn_end':
      onTurnEnd(message.reason);
      break;
    case 'expired':
      // the pass this page was opened with ran out (a tab from hours ago): the relay closes the socket
      expired = true;
      stopCamera();
      setState('lost');
      bar.expired();
      break;
  }
}

function startTurn() {
  // a new turn: everything of the previous one goes (the decoded track stays: it is most likely this turn's)
  result = null;
  shareFile = null;
  fetched = null;
  download = { bytes: 0, total: 0, started: 0, first: 0, trace: [] };
  for (const key of Object.keys(timing)) delete timing[key];
  ended = false;
  streaming = false;
  lastStage = '';
  setFlash(0, 0);
  stopTrack(0.5);
  tapeReset();
  setState('you');
  bar.turn();
}

function onStage(message) {
  if (message.stage === 'flash') {
    if (state !== 'you') return;
    // the flash: the window becomes the main screen, sent frame by frame, and the track starts here,
    // from standstill. The white follows the room's flash from here: up with the flash-in, held while the
    // room waits for the capture's depth, down with the flash-out (the stage messages below)
    setState('transform');
    laughMuted = true;      // until the track has stopped
    flashAt = performance.now();
    bar.flash();
    tapeReset();
    startTrack(0);
    return;
  }
  // the sequence's stages with their progress: the white, the tape and the bar follow them as the room does
  if (state !== 'transform') return;
  const p = message.progress || 0;
  lastStage = `${message.stage} ${p.toFixed(2)}`;
  // `value` is the room's eased progress, the white as the room draws it (an older SMILE sends the progress only)
  const value = 'value' in message ? message.value : p;
  if (message.stage === 'flash_in') {
    setFlash(value, 0.25);
    setLight(Math.max(value, lightNow()), 0.25);
    if (p >= 1) {
      setTimeout(startStreaming, 250);   // under full white the window becomes the main screen
      bar.volume();
    }
  } else if (message.stage === 'flash_out') {
    bar.volume();     // the white was full (a flash-in message at 1 is not always sent: the room's stage moves on first)
    // the window is the main screen now and the room's flash out is in its frames (the capture appearing from
    // the white by its depth): nothing of the page's own white over it; the light behind the frame goes down with it
    startStreaming();
    setFlash(0, 0.25);
    setLight(1 - value, 0.25);
  } else {
    setFlash(0, 0.25);
    setLight(0, 0.25);
  }
  tapeStage(message.stage, p);
  bar.stage(message.stage, p);
}

function startStreaming() {
  streaming = true;
  document.body.classList.add('colour');     // the main screen is in colour, the phone's own image was black and white
}

function onImage(data) {
  const bytes = new Uint8Array(data);
  if (bytes[0] !== SHOW && bytes[0] !== LAST) return;
  // every frame is answered, drawn or not: the relay sends the next one only then, so a slow phone gets
  // fewer frames instead of frames (and everything behind them) seconds late
  send({ type: 'seen' });
  if (state !== 'transform' || !streaming || ended) return;
  const last = bytes[0] === LAST;
  const blob = new Blob([bytes.subarray(1)], { type: 'image/jpeg' });
  const url = URL.createObjectURL(blob);
  const img = new Image();
  img.onload = () => {
    if (last) fadeToLast(img);
    else viewCtx.drawImage(img, 0, 0, W, H);
    URL.revokeObjectURL(url);
  };
  img.src = url;
}

function fadeToLast(img) {
  // the turn's last frame is the result without the caption: the frame shown now fades into it over the
  // caption's fade time, and it stands until the video (the room's bloom and fade are not shown)
  ended = true;
  beforeCtx.drawImage(view, 0, 0);
  const started = performance.now();
  const step = () => {
    const t = Math.min(1, (performance.now() - started) / (CAPTION_FADE * 1000));
    viewCtx.drawImage(img, 0, 0, W, H);
    if (t < 1) {
      viewCtx.globalAlpha = 1 - t;
      viewCtx.drawImage(before, 0, 0);
      viewCtx.globalAlpha = 1;
      requestAnimationFrame(step);
    }
  };
  step();
}

function onTurnEnd(reason) {
  if (reason === 'done') {
    // the sequence is over: the last frame stands; the video takes over when it is here (often already)
    timing.done = performance.now();
    setState('done');
    if (result && !resultUrl(result)) bar.failed(); else bar.done(!!shareFile);
    stopCamera();
    if (result) finishWithVideo();
  } else {
    // missed, lost, reset: back in the line, at the back
    stopTrack(1);
    if (reason === 'lost') lostYou();
    else { setState('waiting'); bar.missed(); }
    join();
  }
}

// THE RESULT: the saved video, uploaded as soon as it is built, usually while the room still shows the
// result; fetched once (for the player and the share sheet). When the turn is over and the file is here,
// the live track winds down to a stop and the video plays with its own sound.
function onResult(message) {
  timing.result = performance.now();
  result = message;
  if (!resultUrl(message)) {
    // no files: the visitor was not kept, or nothing was saved within SMILE's SAVE_TIMEOUT of the turn's end
    fetched = null;
    if (state === 'done') bar.failed();
    return;
  }
  fetched = prefetchShare(message);
  if (state === 'done') finishWithVideo();
}

async function finishWithVideo() {
  const message = result;
  if (!message) return;
  // the file is downloaded once, alone on the link (nothing else is fetched meanwhile); the track plays on
  // until it is here, then winds down and the video plays from it, and it is the share sheet's copy too
  await fetched;
  if (state !== 'done' || result !== message) return;
  windDown(1.2, () => {
    if (message.grin && state === 'done' && result === message) {
      timing.play = performance.now();
      playVideo(shareFile ? URL.createObjectURL(shareFile) : message.grin);
    }
  });
}

function reportTiming() {
  // seconds from the turn's end to each step, to the relay's log: what a phone waits for, measured there;
  // sent when the file is here and again when the video plays, the later line carrying everything
  if (!timing.done) return;
  const since = (t) => (t ? Math.round((t - timing.done) / 100) / 10 : null);
  send({ type: 'timing', result: since(timing.result), play: since(timing.play), playing: since(timing.playing),
         file: since(timing.file), first: since(download.first), kb: Math.round(download.total / 1024),
         trace: download.trace, agent: navigator.userAgent.slice(0, 60) });
}

function playVideo(url) {
  grinVideo.onplaying = () => {
    if (state === 'done') document.body.classList.add('playing');
    if (!timing.playing) { timing.playing = performance.now(); reportTiming(); }
  };
  grinVideo.src = url;
  grinVideo.muted = false;
  // the element was primed by the "I agree" tap, so sound is allowed; a browser that refuses anyway
  // plays it muted, and a tap on the window turns the sound on
  grinVideo.play().catch(() => { grinVideo.muted = true; grinVideo.play().catch(() => {}); });
}

// THE CAMERA: the device, drawn into the frame on every display frame (with the halftone from what the frame
// shows) and sent to SMILE during the turn
let sending = false;    // a frame is being encoded
let lastSend = 0;

async function startCamera() {
  // the camera, or false: the page asks as it loads, and asks again inside the agree tap when there is no
  // camera by then, without waiting for the first ask (on iOS an ask made outside a tap may never be answered:
  // no prompt, no refusal; a request made in a tap is the one every browser treats best). Whichever ask gets
  // a camera first keeps it, the other's is stopped
  try {
    const got = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 1600 } }, audio: false });
    if (stream) { for (const t of got.getTracks()) t.stop(); return true; }
    stream = got;
    cam.srcObject = stream;
    await cam.play().catch(() => {});
    return true;
  } catch (error) {
    return false;
  }
}

function stopCamera() {
  if (stream) for (const t of stream.getTracks()) t.stop();
  stream = null;
}

function cover(sw, sh, dw, dh) {
  // a source of sw x sh scaled to cover dw x dh, the middle kept: its size then
  const k = Math.max(dw / sw, dh / sh);
  return { w: sw * k, h: sh * k };
}

function draw() {
  requestAnimationFrame(draw);
  drawCamera();
  drawHalftone();
}

function drawCamera() {
  if (!stream || cam.readyState < 2) return;
  if (state === 'done' || (state === 'transform' && streaming)) return;
  // the camera as a 4:5 frame: scaled to cover, the middle kept, mirrored like a mirror
  const vw = cam.videoWidth, vh = cam.videoHeight;
  if (!vw || !vh) return;
  const { w, h } = cover(vw, vh, W, H);
  frameCtx.save();
  frameCtx.translate(W, 0);
  frameCtx.scale(-1, 1);
  frameCtx.drawImage(cam, (W - w) / 2, (H - h) / 2, w, h);
  frameCtx.restore();

  if (state === 'you') {
    viewCtx.drawImage(frame, 0, 0);
    drawFaces();
    sendFrame();
  } else if (state === 'transform') {
    viewCtx.drawImage(frame, 0, 0);          // the flash-in: the image as it was, sharp, going white
  } else {
    // waiting, or no connection: the whole image in pixels, the block size from the place in line
    pixelate(0, 0, W, H, eye.block || 64, viewCtx);
  }
}

function pixelate(x, y, w, h, block, ctx) {
  if (block <= 1) { ctx.drawImage(frame, x, y, w, h, x, y, w, h); return; }
  const sw = Math.max(1, Math.round(w / block)), sh = Math.max(1, Math.round(h / block));
  small.width = sw; small.height = sh;
  smallCtx.drawImage(frame, x, y, w, h, 0, 0, sw, sh);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(small, 0, 0, sw, sh, x, y, w, h);
  ctx.imageSmoothingEnabled = true;
}

function drawFaces() {
  // the whole image in pixels as long as the face is not right: the subject's distance from good sets the
  // block, no face is the coarsest; a good face is sharp. Nothing else is drawn: the charge is the ring light
  const subject = faces ? subjectFace(faces) : null;
  const distance = subject ? subject.distance : 1;
  const block = 1 + Math.round(distance * MAX_BLOCK);
  if (block > 1) pixelate(0, 0, W, H, block, viewCtx);
}

function sendFrame() {
  const now = performance.now();
  if (sending || now - lastSend < 1000 / FPS) return;
  if (!ws || ws.readyState !== WebSocket.OPEN || ws.bufferedAmount > 200000) return;
  sending = true;
  lastSend = now;
  frame.toBlob(async (blob) => {
    try {
      if (blob && ws.readyState === WebSocket.OPEN) {
        const bytes = new Uint8Array(await blob.arrayBuffer());
        const message = new Uint8Array(bytes.length + 1);
        message[0] = FRAME;
        message.set(bytes, 1);
        ws.send(message);
      }
    } finally {
      sending = false;
    }
  }, 'image/jpeg', QUALITY);
}

// THE SHARING: the grin through the share sheet, with the caption as the share text (and on the clipboard,
// because Instagram drops it); the file is fetched when the result comes (prefetchShare), because a share
// sheet has to open right on the tap, a phone refuses one that comes seconds later; a download where there
// is no sheet.
// the video's download as it comes in: bytes so far, the total, when it started, the first byte, and the bytes
// at every second (KB), for the readout and the timing line: a wait before the first byte, a trickle, bursts
// with pauses, or the whole file early and the page late are four different problems
let download = { bytes: 0, total: 0, started: 0, first: 0, trace: [] };

async function prefetchShare(message) {
  shareFile = null;
  const url = resultUrl(message);
  if (!url) return;
  download = { bytes: 0, total: 0, started: performance.now(), first: 0, trace: [] };
  try {
    const response = await fetch(url);
    download.total = Number(response.headers.get('content-length')) || 0;
    const blob = await readMeasured(response);
    if (result === message) { shareFile = new File([blob], fileName(message, url), { type: blob.type }); timing.file = performance.now(); reportTiming(); }
  } catch (error) { /* the tap then fetches it */ }
  if (state === 'done' && result === message) bar.offer(!!shareFile);
}

async function readMeasured(response) {
  // the body read chunk by chunk, so the readout and the timing line see the download as it happens
  if (!response.body || !response.body.getReader) return response.blob();
  const reader = response.body.getReader();
  const chunks = [];
  const sampler = setInterval(() => { if (download.trace.length < 120) download.trace.push(Math.round(download.bytes / 1024)); }, 1000);
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!download.first) download.first = performance.now();
      download.bytes += value.byteLength;
      chunks.push(value);
    }
  } finally { clearInterval(sampler); }
  return new Blob(chunks, { type: response.headers.get('content-type') || 'video/mp4' });
}

function fileName(message, url) {
  // the file the visitor keeps or shares is the artist's: smile_by_<the credit's name>, with the file's own
  // extension, whatever the relay calls it
  const name = String(message.name || '').toLowerCase().normalize('NFD').replace(ACCENTS, '')
    .replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
  const ext = (String(url).match(/\.[a-z0-9]+$/i) || ['.mp4'])[0];
  return (name ? `smile_by_${name}` : 'smile') + ext;
}

async function shareSmile() {
  if (!result) return;
  const text = `${result.name || ''} ${result.caption || ''} ${TAGS}`.trim();
  try { navigator.clipboard.writeText(text).catch(() => {}); } catch (error) { /* no clipboard: the share text is all */ }
  try {
    if (!shareFile && fetched) await fetched;
    if (!shareFile) await prefetchShare(result);
    if (shareFile && navigator.canShare && navigator.canShare({ files: [shareFile] })) {
      try {
        await navigator.share({ files: [shareFile], text });
      } catch (error) {
        // closed without sharing: nothing, the button stays for another try (a download here would put
        // iOS's "download or view" in the way, and "view" leaves the page, which reloads)
        if (error.name !== 'AbortError') throw error;
      }
      return;
    }
  } catch (error) { /* no share sheet, or one refused (opened too late after the tap): download instead */ }
  // from the copy on the phone when there is one: the relay deletes the file minutes after the turn
  const a = document.createElement('a');
  a.href = shareFile ? URL.createObjectURL(shareFile) : resultUrl(result);
  a.download = fileName(result, resultUrl(result));
  a.click();
  if (shareFile) setTimeout(() => URL.revokeObjectURL(a.href), 60000);
}

// THE JOINING: the page asks for the camera as it opens (THE BOOT) and shows it in the frame, in pixels,
// under the terms; the bar under the frame is the agree button: the one tap is the consent, the sound's
// unlock and the joining of the line, and the bar goes black for the wait. Without a camera the terms go,
// the bar says so and nothing can be tapped.
async function agree() {
  unlockAudio();
  const ok = stream || await startCamera();     // the ask in the tap, when the load's got nothing
  $('terms').hidden = true;
  if (!ok) { noCamera(); return; }
  setState('waiting');
  bar.wait(eye);
  join();
}

function noCamera() {
  setState('nocam');
  bar.nocam();
}

// THE HALFTONE: the page behind the frame is the picture the frame shows as a field of smileys, black and
// white: one cell of CELL css pixels per face, its size from the brightness there, a dot in the dark, a ring,
// and a face with eyes and a smile in the light, breathing a little so the field moves while the picture
// stands still. The source is the camera (mirrored, as the frame), the main screen while it streams, the last
// frame while it stands, the video when it plays; nothing before the camera is on.
// One WebGL quad: the picture goes up as a texture and the fragment shader draws the faces, one draw every
// other display frame and nothing per cell on the CPU, so a weak phone keeps up; without WebGL the
// background stays black.
const half = { gl: null, cols: 1, rows: 1, odd: false, source: null, u: {} };

const HALFTONE_VERT = `
attribute vec2 p;
varying vec2 v;
void main() { v = p; gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }`;

const HALFTONE_FRAG = `
precision mediump float;
varying vec2 v;
uniform sampler2D source;
uniform vec2 grid;          // cells across and down
uniform vec2 scale;         // the source fitted to cover the page: source uv = page uv * scale + shift
uniform vec2 shift;
uniform float time;

// a line of half width w at distance d, with a soft edge
float line(float d, float w) { return 1.0 - smoothstep(w - 0.05, w + 0.05, d); }

void main() {
  vec2 cell = floor(v * grid);
  vec3 c = texture2D(source, (cell + 0.5) / grid * scale + shift).rgb;
  float luma = dot(c, vec3(0.299, 0.587, 0.114));
  float value = clamp(luma + 0.08 * sin(time * 0.7 + cell.x * 0.3 + cell.y * 0.2), 0.0, 1.0);
  vec2 p = fract(v * grid) * 2.0 - 1.0;      // -1..1 across the cell
  float r = 0.12 + 0.76 * value;             // the face's radius: a dot in the dark, a whole face in the light
  float d = length(p);
  float ink;
  if (r < 0.3) ink = 1.0 - smoothstep(r, r + 0.08, d);     // the smallest: a dot
  else ink = line(abs(d - r), 0.09);                        // a ring
  if (r > 0.5) {
    // the face: two eyes, and a smile (the lower arc of a circle set a little above the middle)
    float eye = min(length(p - vec2(-0.36, 0.28) * r), length(p - vec2(0.36, 0.28) * r));
    ink = max(ink, 1.0 - smoothstep(0.1 * r, 0.1 * r + 0.08, eye));
    float arc = abs(length(p - vec2(0.0, 0.12 * r)) - 0.6 * r);
    ink = max(ink, line(arc, 0.07) * step(p.y, -0.12 * r));
  }
  gl_FragColor = vec4(vec3(0.4 * ink), 1.0);
}`;

function initHalftone() {
  half.gl = null;
  let gl = null;
  try { gl = halftone.getContext('webgl', { alpha: false, antialias: false, depth: false, stencil: false }); } catch (error) { gl = null; }
  if (!gl) return;
  const shader = (type, text) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, text);
    gl.compileShader(s);
    return s;
  };
  const program = gl.createProgram();
  gl.attachShader(program, shader(gl.VERTEX_SHADER, HALFTONE_VERT));
  gl.attachShader(program, shader(gl.FRAGMENT_SHADER, HALFTONE_FRAG));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) return;
  gl.useProgram(program);
  const quad = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quad);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
  const p = gl.getAttribLocation(program, 'p');
  gl.enableVertexAttribArray(p);
  gl.vertexAttribPointer(p, 2, gl.FLOAT, false, 0, 0);
  // the source texture: any size (no power of two), so clamped and without mipmaps
  half.source = gl.createTexture();
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, half.source);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  for (const name of ['source', 'grid', 'scale', 'shift', 'time']) half.u[name] = gl.getUniformLocation(program, name);
  gl.uniform1i(half.u.source, 0);
  half.gl = gl;
  sizeHalftone();
}

function sizeHalftone() {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  half.cols = Math.max(1, Math.round(window.innerWidth / CELL));
  half.rows = Math.max(1, Math.round(window.innerHeight / CELL));
  halftone.width = Math.round(window.innerWidth * dpr);
  halftone.height = Math.round(window.innerHeight * dpr);
  const gl = half.gl;
  if (!gl) return;
  gl.viewport(0, 0, halftone.width, halftone.height);
  gl.uniform2f(half.u.grid, half.cols, half.rows);
}

function halftoneSource() {
  if (state === 'done') return (document.body.classList.contains('playing') && grinVideo.readyState >= 2) ? grinVideo : view;
  if (state === 'transform' && streaming) return view;
  if (stream && cam.readyState >= 2) return frame;
  return null;
}

function drawHalftone() {
  const gl = half.gl;
  if (!gl) return;
  half.odd = !half.odd;
  if (half.odd) return;      // every other display frame is enough
  const src = halftoneSource();
  const sw = src ? (src.videoWidth || src.width) : 0, sh = src ? (src.videoHeight || src.height) : 0;
  if (!sw || !sh) {
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    return;
  }
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, half.source);
  try { gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src); } catch (error) { return; }   // a video frame not yet decodable
  // the source scaled to cover the page, the middle kept
  const { w, h } = cover(sw, sh, window.innerWidth, window.innerHeight);
  const sx = window.innerWidth / w, sy = window.innerHeight / h;
  gl.uniform2f(half.u.scale, sx, sy);
  gl.uniform2f(half.u.shift, (1 - sx) / 2, (1 - sy) / 2);
  gl.uniform1f(half.u.time, performance.now() / 1000);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

// THE TERMS: the heading and the list each at the size that makes its longest line span the frame, both
// smaller by the same factor when they would not fit its height; measured at 10 px and scaled. Nothing to do
// while the terms are gone or the page is hidden (a turned phone)
function fitTerms() {
  const terms = $('terms'), box = $('terms-box');
  if (terms.hidden) return;
  const style = getComputedStyle(terms);
  const width = terms.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
  const height = terms.clientHeight - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom);
  if (!(width > 0 && height > 0)) return;
  const head = box.querySelector('h2'), list = box.querySelector('ol');
  box.style.setProperty('--terms-head', '10px');
  box.style.setProperty('--terms', '10px');
  let headSize = 10 * width / head.scrollWidth, listSize = 10 * width / list.scrollWidth;
  const needed = (head.scrollHeight + 3) * headSize / 10 + list.scrollHeight * listSize / 10;   // the margin under the heading is 0.3 em
  if (needed > height) { headSize *= height / needed; listSize *= height / needed; }
  box.style.setProperty('--terms-head', `${headSize.toFixed(2)}px`);
  box.style.setProperty('--terms', `${listSize.toFixed(2)}px`);
}

// THE READOUT, with ?debug in the address: the state, the sound and the download, for testing on a phone
function readout() {
  const src = track.source;
  debugLine.textContent = [
    `state ${state}  stage ${lastStage}`,
    `audio ${audioCtx ? audioCtx.state : 'none'}  laughs ${laughs.length} muted ${laughMuted}`,
    `track ${track.id || '-'} ${track.buffer ? 'decoded' : 'no buffer'} ${track.playing ? 'playing' : 'stopped'} ${track.error}`,
    `rate ${src ? src.playbackRate.value.toFixed(2) : '-'} target ${tape.target.toFixed(2)}  gain ${musicGain ? musicGain.gain.value.toFixed(2) : '-'}`,
    `filter ${musicFilter ? Math.round(musicFilter.frequency.value) : '-'} Hz q ${musicFilter ? musicFilter.Q.value.toFixed(1) : '-'}`,
    `video ${grinVideo.src ? (grinVideo.paused ? 'paused' : 'playing') : '-'} ${grinVideo.muted ? 'muted' : 'sound'}`,
    downloadLine(),
    `halftone ${half.gl ? 'webgl' : 'none'} ${half.cols}x${half.rows}`,
  ].join('\n');
}

function downloadLine() {
  if (!download.started) return 'download -';
  const now = performance.now();
  const seconds = ((shareFile ? timing.file : now) - download.started) / 1000;
  const first = download.first ? `first byte ${((download.first - download.started) / 1000).toFixed(1)} s` : 'no byte yet';
  return `download ${Math.round(download.bytes / 1024)}/${Math.round(download.total / 1024)} KB  ${seconds.toFixed(1)} s  ${first}`;
}

// THE BOOT: everything that runs as the page loads, in one place, after everything it needs
$('window').onclick = () => {
  // a tap on the frame turns the video's sound on where the browser refused it (playVideo)
  if (state === 'done' && grinVideo.src && grinVideo.muted) {
    grinVideo.muted = false;
    grinVideo.play().catch(() => {});
  }
};
share.onclick = () => {
  if (share.classList.contains('off')) return;
  if (state === 'idle') agree(); else shareSmile();
};
share.onkeydown = (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); share.onclick(); } };   // a keyboard, on a desktop
halftone.addEventListener('webglcontextlost', (event) => { event.preventDefault(); half.gl = null; });
halftone.addEventListener('webglcontextrestored', initHalftone);
window.addEventListener('resize', sizeHalftone);
window.addEventListener('resize', fitTerms);
if (document.fonts && document.fonts.ready) document.fonts.ready.then(fitTerms);     // the font's width, not the fallback's
if (location.search.includes('debug')) {
  debugLine.hidden = false;
  setInterval(readout, 250);
}
initHalftone();
setState('idle');
fitTerms();
bar.agree(false);                                                                   // the bar opens white, still
startCamera().then((ok) => { if (ok && state === 'idle') bar.agree(true); });     // the camera allowed: the button moves
connect();
draw();
