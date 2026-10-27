/* SMILE Online, the phone page. The grid shows the newest portraits; the window in its middle is the
   camera: pixelated with the reason while you wait; when it is your turn, your live video, in pixels as
   long as the face is not right, the worse the coarser, sharp when it is (the face data comes from SMILE,
   the drawing happens here, so only the judging lags; nothing is drawn on the image), and the charge is
   the ring light: the page goes white behind the grid as it fills, quickly at first, full white at the flash. From the flash the window is the main screen of the installation
   as it is rendered, frame by frame, to the end of its sequence, while the white fades with the
   installation's flash, and the visitor's track plays from the flash as in the room: on the tape, spun up
   from standstill at the flash, muffled and opening up through the transformation, from the room's own
   stage progress (the Max patch's curves), within what a phone speaker can do: the filter never closes
   below PHONE_FLOOR and the tape never runs under 0.3 (it is silent instead), as the patch has it. The track
   is named at the turn's start and fetched and decoded during the judging; it plays through Web Audio, a
   media element would be silenced at the tape's low speeds. With ?debug in the address the page shows its sound state in a corner. The turn is over after the fade, but the track plays on until the saved video is there (the share
   button appears when it is here; it is fetched while the room still shows the result, so it mostly is):
   then it winds down to a stop and the video takes over, looping with its own sound. The room's laughter samples play at random from the moment of consent,
   panned about, while nothing else sounds.
   The camera is drawn as a mirror and SENT as a mirror too: the installation and the portrait then
   show what you saw on the phone, not its mirror image.
   Messages: see relay/server.py. */

'use strict';

const FRAME = 0;        // a binary message: our camera frame to SMILE
const SHOW = 1;         // a binary message: a frame of the main screen from SMILE
const W = 640, H = 800; // the frame we send and draw, 4:5 like the installation's image
const FPS = 15;         // frames a second to SMILE during the turn
const QUALITY = 0.6;    // JPEG
const MAX_BLOCK = 32;   // a face's pixel size at distance 1 (the installation's face_effect)
const TAGS = '#smile #oostrik';   // the share text, until the real tags are decided

const $ = (id) => document.getElementById(id);
const cam = $('cam'), view = $('view'), grinVideo = $('grin'), portraitImg = $('portrait'), light = $('light');
const statusLine = $('status'), flash = $('flash'), share = $('share');
const frame = document.createElement('canvas');     // the camera as we send it: mirrored
const small = document.createElement('canvas');     // for the pixelation
view.width = frame.width = W;
view.height = frame.height = H;
const viewCtx = view.getContext('2d'), frameCtx = frame.getContext('2d'), smallCtx = small.getContext('2d');

let ws = null;
let state = 'idle';     // idle, waiting, you, transform, done, asleep, lost, nocam
let eye = {};           // the last state message from the relay
let faces = null;       // the last face message during the turn
let stream = null;      // the camera
let sending = false;    // a frame is being encoded
let lastSend = 0;
let result = null;      // the result message: urls and caption, after the turn
let shareFile = null;   // the video as a File, fetched when the result comes, so the share sheet opens at once
let turnWindow = 10;
let audioCtx = null, musicGain = null, musicFilter = null;    // the track's chain: source -> low-pass filter -> volume
let track = { id: '', buffer: null, source: null, playing: false, error: '' };    // this turn's track, decoded, and its player
let pendingTrack = null;    // a track named before the sound was unlocked: fetched at the unlock
let lastStage = '';
const timing = {};          // moments of the turn's end (ms since the page loaded): what took how long, for the relay's log
let flashAt = 0;            // performance.now() of the flash: the track starts there
let streaming = false;      // the window shows the main screen: from full white on, the flash-in is the phone's own image going white
let laughs = [];            // the laughter samples' urls, from the relay's library
const laughBuffers = new Map();
let laughMuted = false;     // from the flash until the track has stopped: the laughter is for the waiting, not over the music
let laughTimer = null;

// THE GRID: the newest stills first, built once from the relay's first list and left alone after that
// (empty cells until the relay answers)
let gridBuilt = false;
function setGrid(stills) {
  if (gridBuilt) return;
  gridBuilt = stills.length > 0;
  const grid = $('grid');
  for (const img of grid.querySelectorAll('img.still')) img.remove();
  const cells = Math.max(12, stills.length);
  for (let i = 0; i < cells; i++) {
    const img = document.createElement('img');
    img.className = 'still';
    if (stills[i]) img.src = stills[i];
    grid.appendChild(img);
  }
}

// THE STATE
function setState(next) {
  state = next;
  const video = document.body.classList.contains('video');
  document.body.className = next;
  if (video) document.body.classList.add('video');
  if (next === 'done') document.body.classList.add('colour');
  if (next !== 'you') faces = null;
  if (next !== 'you' && next !== 'transform') setLight(0, 0.25);
  if (next !== 'done') { grinVideo.pause(); document.body.classList.remove('playing'); }
}

function setStatus(text) {
  statusLine.textContent = text || '';
}

// THE RING LIGHT: the white behind the grid, 0..1, reached in `seconds`
function setLight(value, seconds) {
  const duration = `${seconds}s`;
  light.style.transitionDuration = duration;
  for (const img of document.querySelectorAll('#grid img.still')) img.style.transitionDuration = duration;
  document.documentElement.style.setProperty('--light', String(value));
}

// THE FLASH over the window, 0..1, reached in `seconds`: the room's flash as its progress says
function setFlash(value, seconds) {
  flash.style.transitionDuration = `${seconds}s`;
  flash.style.opacity = String(value);
}

// THE RELAY
function connect() {
  const url = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/phone';
  ws = new WebSocket(url);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => {
    if (stream && state !== 'idle') ws.send(JSON.stringify({ type: 'join' }));
  };
  ws.onmessage = (event) => {
    if (typeof event.data === 'string') onMessage(JSON.parse(event.data));
    else onImage(event.data);
  };
  ws.onclose = () => {
    if (state === 'you' || state === 'transform' || state === 'waiting') { setState('lost'); setStatus('lost you'); }
    setTimeout(connect, 2000);
  };
  ws.onerror = () => ws.close();
}

function onMessage(message) {
  switch (message.type) {
    case 'grid':
      setGrid(message.stills);
      break;
    case 'state':
      eye = message;
      if (state === 'idle' || state === 'nocam') {
        setStatus(message.status);
      } else if (state === 'waiting' || state === 'asleep' || state === 'lost') {
        setState(message.eye === 'asleep' ? 'asleep' : 'waiting');
        setStatus(message.status);
      }
      break;
    case 'turn':
      turnWindow = message.window || 10;
      result = null;
      for (const key of Object.keys(timing)) delete timing[key];
      document.body.classList.remove('video');
      stopTrack(0.5);           // the decoded track stays: it is most likely this turn's
      tapeReset();
      setState('you');
      setStatus('');
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
  }
}

function onStage(message) {
  if (message.stage === 'flash') {
    if (state !== 'you') return;
    // the flash: the window becomes the main screen, sent frame by frame, and the track starts here,
    // from standstill. The white follows the room's flash from here: up with the flash-in, held while the
    // room waits for the capture's depth, down with the flash-out (the stage messages below)
    setState('transform');
    setStatus('');
    streaming = false;      // the phone's own image goes white first (the room's marks are not for the phone)
    laughMuted = true;      // until the track has stopped
    flashAt = performance.now();
    tapeReset();
    startTrack(0);
    return;
  }
  // the sequence's stages with their progress: the white and the tape follow them as the room does
  if (state !== 'transform') return;
  const p = message.progress || 0;
  lastStage = `${message.stage} ${p.toFixed(2)}`;
  if (message.stage === 'flash_in') {
    const white = p * p;                                // the room's flash-in eases in (quad)
    setFlash(white, 0.25);
    setLight(Math.max(white, lightNow()), 0.25);
    if (p >= 1) setTimeout(startStreaming, 250);   // under full white the window becomes the main screen
  } else if (message.stage === 'flash_out') {
    startStreaming();
    const white = 1 - Math.sin(p * Math.PI / 2);        // and eases out (sine)
    setFlash(white, 0.25);
    setLight(white, 0.25);
  } else {
    setFlash(0, 0.25);
    setLight(0, 0.25);
  }
  tapeStage(message.stage, p);
}

function startStreaming() {
  streaming = true;
  document.body.classList.add('colour');     // the main screen is in colour, the phone's own image was black and white
}

function lightNow() {
  return parseFloat(document.documentElement.style.getPropertyValue('--light')) || 0;
}

function onImage(data) {
  const bytes = new Uint8Array(data);
  if (bytes[0] !== SHOW) return;
  // every frame is answered, drawn or not: the relay sends the next one only then, so a slow phone gets
  // fewer frames instead of frames (and everything behind them) seconds late
  if (ws && ws.readyState === WebSocket.OPEN) ws.send('{"type":"seen"}');
  if (state !== 'transform' || !streaming) return;
  const blob = new Blob([bytes.subarray(1)], { type: 'image/jpeg' });
  const url = URL.createObjectURL(blob);
  const img = new Image();
  img.onload = () => {
    viewCtx.drawImage(img, 0, 0, W, H);
    URL.revokeObjectURL(url);
  };
  img.src = url;
}

function onTurnEnd(reason) {
  if (reason === 'done') {
    // the sequence is over: the last frame stands; the video takes over when it is here (often already)
    timing.done = performance.now();
    setState('done');
    setStatus('');
    stopCamera();
    setShare(result);
    if (result) finishWithVideo();
  } else {
    // missed, lost, reset: back in the line, at the back
    stopTrack(1);
    setState(reason === 'lost' ? 'lost' : 'waiting');
    setStatus(reason === 'lost' ? 'lost you' : 'missed you');
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'join' }));
  }
}

// THE RESULT: the saved video, uploaded as soon as it is built, usually while the room still shows the
// result; fetched once (for the player and the share sheet). When the turn is over and the file is here,
// the live track winds down to a stop and the video plays with its own sound.
let fetched = null;
function onResult(message) {
  timing.result = performance.now();
  result = message;
  document.body.classList.toggle('video', !!result.grin);
  if (result.still || result.portrait) portraitImg.src = result.still || result.portrait;
  fetched = prefetchShare(message);
  if (state === 'done') finishWithVideo();
}

async function finishWithVideo() {
  const message = result;
  setShare(message);
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
  // seconds from the turn's end to each step, to the relay's log: what a phone waits for, measured there
  if (!ws || ws.readyState !== WebSocket.OPEN || !timing.done) return;
  const since = (t) => (t ? Math.round((t - timing.done) / 100) / 10 : null);
  ws.send(JSON.stringify({ type: 'timing', result: since(timing.result), play: since(timing.play), playing: since(timing.playing),
                           file: since(timing.file), agent: navigator.userAgent.slice(0, 60) }));
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

$('window').onclick = () => {
  if (state === 'done' && grinVideo.src && grinVideo.muted) {
    grinVideo.muted = false;
    grinVideo.play().catch(() => {});
  }
};

function setShare(result) {
  // the button appears when the video is here on the phone; nothing before, nothing without one
  const ready = !!(result && (result.grin || result.portrait) && shareFile);
  share.disabled = !ready;
  share.hidden = !ready;
}

// THE SOUND: browsers only play sound after a tap, so the "I agree" tap wakes the audio context (the
// track and the laughter play through it) and primes the video element with a moment of silence, so it
// may play with sound later without another tap.
function silentWav() {
  const samples = 441, header = 44, bytes = new Uint8Array(header + samples * 2), view = new DataView(bytes.buffer);
  const text = (at, s) => { for (let i = 0; i < s.length; i++) bytes[at + i] = s.charCodeAt(i); };
  text(0, 'RIFF'); view.setUint32(4, 36 + samples * 2, true); text(8, 'WAVE'); text(12, 'fmt ');
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, 44100, true); view.setUint32(28, 88200, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  text(36, 'data'); view.setUint32(40, samples * 2, true);
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

function flushPendingTrack() {
  if (pendingTrack) { const { id, url } = pendingTrack; prepareTrack(id, url); }
}

// THE TRACK, through Web Audio: fetched and decoded when named, started at the flash (or as soon as it is
// decoded, that far in), played on the tape, wound down when the video is there
async function prepareTrack(id, url) {
  if (track.id === id) return;
  if (!audioCtx) { pendingTrack = { id, url }; return; }     // fetched at the unlock
  pendingTrack = null;
  stopTrack(0);
  track = { id, buffer: null, source: null, playing: false, error: '' };
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`fetch ${response.status}`);
    const buffer = await audioCtx.decodeAudioData(await response.arrayBuffer());
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

function stopTrack(seconds) {
  const source = track.source;
  track.source = null;
  track.playing = false;
  laughMuted = false;
  if (!source || !audioCtx) return;
  const t = audioCtx.currentTime;
  try {
    if (seconds > 0) {
      musicGain.gain.cancelScheduledValues(t);
      musicGain.gain.setValueAtTime(musicGain.gain.value, t);
      musicGain.gain.linearRampToValueAtTime(0, t + seconds);
      source.stop(t + seconds);
    } else {
      source.stop(t);
    }
  } catch (error) { /* already stopped */ }
}

// the wind-down: the speed from where it is to a stop over `seconds` (the patch's fade), then `done`
function windDown(seconds, done) {
  const source = track.source;
  if (!source || !audioCtx) { stopTrack(0); done(); return; }
  const t = audioCtx.currentTime;
  tape.target = 0;
  source.playbackRate.cancelScheduledValues(t);
  source.playbackRate.setValueAtTime(source.playbackRate.value, t);
  source.playbackRate.linearRampToValueAtTime(0, t + seconds);
  musicGain.gain.cancelScheduledValues(t);
  musicGain.gain.setValueAtTime(musicGain.gain.value, t + seconds * 0.8);
  musicGain.gain.linearRampToValueAtTime(0, t + seconds);
  track.source = null;
  track.playing = false;
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
    case 'grin': case 'timeout':
      tape.target = 1; tape.freq = 20000; tape.q = 1;
      break;
    case 'bloom': case 'fade':
      // the room winds down here; the phone's track plays on until the video is there (onResult)
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

// THE READOUT, with ?debug in the address: what the sound does, for testing on a phone
const debugLine = $('debug');
if (location.search.includes('debug')) {
  debugLine.hidden = false;
  setInterval(() => {
    const src = track.source;
    debugLine.textContent = [
      `state ${state}  stage ${lastStage}`,
      `audio ${audioCtx ? audioCtx.state : 'none'}  laughs ${laughs.length} muted ${laughMuted}`,
      `track ${track.id || '-'} ${track.buffer ? 'decoded' : 'no buffer'} ${track.playing ? 'playing' : 'stopped'} ${track.error}`,
      `rate ${src ? src.playbackRate.value.toFixed(2) : '-'} target ${tape.target.toFixed(2)}  gain ${musicGain ? musicGain.gain.value.toFixed(2) : '-'}`,
      `filter ${musicFilter ? Math.round(musicFilter.frequency.value) : '-'} Hz q ${musicFilter ? musicFilter.Q.value.toFixed(1) : '-'}`,
      `video ${grinVideo.src ? (grinVideo.paused ? 'paused' : 'playing') : '-'} ${grinVideo.muted ? 'muted' : 'sound'}`,
    ].join('\n');
  }, 250);
}

// THE LAUGHTER: the room's samples (the relay's library), one every second or two at a random place while
// there is no music (the wall laughs all the time); none over the track or the video
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
      const response = await fetch(url);
      buffer = await audioCtx.decodeAudioData(await response.arrayBuffer());
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

// JOINING
$('join').onclick = () => { $('terms').hidden = false; };
$('agree').onclick = async () => {
  $('terms').hidden = true;
  unlockAudio();
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 1600 } }, audio: false });
    cam.srcObject = stream;
    await cam.play();
    setState('waiting');
    setStatus(eye.status || '');
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'join' }));
  } catch (error) {
    setState('nocam');
    setStatus('no camera, no smile');
  }
};

function stopCamera() {
  if (stream) for (const track of stream.getTracks()) track.stop();
  stream = null;
}

// SHARING: the grin with its caption through the share sheet; the text goes to the clipboard too,
// because Instagram drops it. The file is fetched when the result comes (prefetchShare): a share sheet
// has to open right on the tap, a phone refuses one that comes seconds later.
async function prefetchShare(message) {
  shareFile = null;
  const url = message.grin || message.portrait;
  if (!url) return;
  try {
    const response = await fetch(url);
    const blob = await response.blob();
    if (result === message) { shareFile = new File([blob], url.split('/').pop(), { type: blob.type }); timing.file = performance.now(); reportTiming(); }
  } catch (error) { /* the tap then fetches it */ }
  if (state === 'done' && result === message) setShare(message);
}

share.onclick = async () => {
  if (!result || share.disabled) return;
  const text = `${result.name || ''} ${result.caption || ''} ${TAGS}`.trim();
  try { navigator.clipboard.writeText(text).catch(() => {}); } catch (error) { /* no clipboard: the share text is all */ }
  try {
    if (!shareFile && fetched) await fetched;
    if (!shareFile) await prefetchShare(result);
    if (shareFile && navigator.canShare && navigator.canShare({ files: [shareFile] })) {
      await navigator.share({ files: [shareFile], text });
      return;
    }
  } catch (error) { /* the share sheet was closed, or there is none: download instead */ }
  const a = document.createElement('a');
  a.href = result.grin || result.portrait;
  a.download = '';
  a.click();
};

// DRAWING, on every display frame
function draw() {
  requestAnimationFrame(draw);
  if (!stream || cam.readyState < 2) return;
  if (state === 'done' || (state === 'transform' && streaming)) return;
  // the camera as a 4:5 frame: scaled to cover, the middle kept, mirrored like a mirror
  const vw = cam.videoWidth, vh = cam.videoHeight;
  if (!vw || !vh) return;
  const scale = Math.max(W / vw, H / vh);
  const dw = vw * scale, dh = vh * scale;
  frameCtx.save();
  frameCtx.translate(W, 0);
  frameCtx.scale(-1, 1);
  frameCtx.drawImage(cam, (W - dw) / 2, (H - dh) / 2, dw, dh);
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
  const subject = faces ? (faces.faces || []).find((face) => face.subject) : null;
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

setGrid([]);
setState('idle');
connect();
draw();
