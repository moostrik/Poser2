/* SMILE Online, the phone page. The grid shows the newest portraits; the window in its middle is the
   camera: pixelated with the reason while you wait, sharp and judged when it is your turn (the face
   data comes from SMILE, the drawing happens here on the live video, so only the judging lags), then
   the transformation as SMILE sends its images, the grin with its caption, and the portrait stays.
   Messages: see relay/server.py. */

'use strict';

const FRAME = 0;        // a binary message: our camera frame to SMILE
const SHOW = 1;         // a binary message: an image from SMILE for the window
const W = 640, H = 800; // the frame we send and draw, 4:5 like the installation's image
const FPS = 15;         // frames a second to SMILE during the turn
const QUALITY = 0.6;    // JPEG
const MAX_BLOCK = 32;   // a face's pixel size at distance 1 (the installation's face_effect)
const TAGS = '#smile #oostrik';   // the share text, until the real tags are decided

const $ = (id) => document.getElementById(id);
const cam = $('cam'), view = $('view'), fill = $('fill'), grinVideo = $('grin'), portraitImg = $('portrait');
const statusLine = $('status'), labels = $('labels'), describe = $('describe'), flash = $('flash');
const frame = document.createElement('canvas');     // the camera as we send it
const small = document.createElement('canvas');     // for the pixelation
view.width = fill.width = frame.width = W;
view.height = fill.height = frame.height = H;
const viewCtx = view.getContext('2d'), fillCtx = fill.getContext('2d'), frameCtx = frame.getContext('2d'), smallCtx = small.getContext('2d');

let ws = null;
let state = 'idle';     // idle, waiting, you, transform, grin, done, asleep, lost, nocam
let eye = {};           // the last state message from the relay
let faces = null;       // the last face message during the turn
let stream = null;      // the camera
let sending = false;    // a frame is being encoded
let lastSend = 0;
let result = null;      // the result message: urls and caption
let turnWindow = 10;

// THE GRID
function setGrid(stills) {
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
  document.body.className = next;
  if (next === 'you') document.body.classList.add('light');
  if (next === 'transform' || next === 'grin' || next === 'done') document.body.classList.add('colour');
  if (next !== 'transform') describe.textContent = '';
  if (next !== 'you') { faces = null; labels.textContent = ''; fillCtx.clearRect(0, 0, W, H); }
}

function setStatus(text) {
  statusLine.textContent = text || '';
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
      setState('you');
      setStatus('');
      break;
    case 'face':
      if (state === 'you') faces = message;
      break;
    case 'stage':
      onStage(message);
      break;
    case 'result':
      result = message;
      if (result.grin) {
        grinVideo.src = result.grin;
        grinVideo.play().catch(() => {});
        setState('grin');
      }
      $('caption-name').textContent = message.name || '';
      $('caption-line').textContent = message.caption || '';
      if (result.still || result.portrait) portraitImg.src = result.still || result.portrait;
      break;
    case 'turn_end':
      onTurnEnd(message.reason);
      break;
  }
}

function onStage(message) {
  if (state !== 'you' && state !== 'transform') return;
  if (message.stage === 'flash') {
    flash.classList.add('on');
    setState('transform');
    setStatus('');
    requestAnimationFrame(() => requestAnimationFrame(() => flash.classList.remove('on')));
  } else if (message.stage === 'describe') {
    document.body.classList.add('describe');
    describe.textContent = message.text || '';
  } else if (message.stage === 'diffusion') {
    document.body.classList.remove('describe');
  }
}

function onImage(data) {
  const bytes = new Uint8Array(data);
  if (bytes[0] !== SHOW || state !== 'transform') return;
  const blob = new Blob([bytes.subarray(1)], { type: 'image/jpeg' });
  const url = URL.createObjectURL(blob);
  const img = new Image();
  img.onload = () => {
    // the transformation is not mirrored: it is the machine's image, not yours
    viewCtx.save();
    viewCtx.translate(W, 0);
    viewCtx.scale(-1, 1);
    viewCtx.drawImage(img, 0, 0, W, H);
    viewCtx.restore();
    URL.revokeObjectURL(url);
  };
  img.src = url;
}

function onTurnEnd(reason) {
  if (reason === 'done' && result) {
    setState('done');
    setStatus('');
    stopCamera();
  } else {
    // missed, lost, reset: back in the line, at the back
    setState(reason === 'lost' ? 'lost' : 'waiting');
    setStatus(reason === 'lost' ? 'lost you' : 'missed you');
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'join' }));
  }
}

// JOINING
$('join').onclick = () => { $('terms').hidden = false; };
$('agree').onclick = async () => {
  $('terms').hidden = true;
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
// because Instagram drops it
$('share').onclick = async () => {
  if (!result) return;
  const text = `${result.name || ''} ${result.caption || ''} ${TAGS}`.trim();
  try { await navigator.clipboard.writeText(text); } catch (error) { /* no clipboard: the share text is all */ }
  try {
    const url = result.grin || result.portrait;
    const response = await fetch(url);
    const blob = await response.blob();
    const file = new File([blob], url.split('/').pop(), { type: blob.type });
    if (navigator.canShare && navigator.canShare({ files: [file] })) {
      await navigator.share({ files: [file], text });
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
  if (state === 'grin' || state === 'done' || state === 'transform') return;
  // the camera as a 4:5 frame: scaled to cover, the middle kept
  const vw = cam.videoWidth, vh = cam.videoHeight;
  if (!vw || !vh) return;
  const scale = Math.max(W / vw, H / vh);
  const dw = vw * scale, dh = vh * scale;
  frameCtx.drawImage(cam, (W - dw) / 2, (H - dh) / 2, dw, dh);

  if (state === 'you') {
    viewCtx.drawImage(frame, 0, 0);
    drawFaces();
    sendFrame();
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
  fillCtx.clearRect(0, 0, W, H);
  labels.textContent = '';
  if (!faces) return;
  const charge = faces.charge || 0;
  for (const face of faces.faces || []) {
    const [x0, y0, x1, y1] = face.box;
    const x = Math.max(0, x0 * W), y = Math.max(0, y0 * H);
    const w = Math.min(W, x1 * W) - x, h = Math.min(H, y1 * H) - y;
    if (w <= 0 || h <= 0) continue;
    // the face in pixels by its distance from good; the subject's box turns to colour with the charge
    const block = 1 + Math.round(face.distance * MAX_BLOCK);
    pixelate(x, y, w, h, block, viewCtx);
    if (face.subject && charge > 0) {
      fillCtx.globalAlpha = charge;
      pixelate(x, y, w, h, block, fillCtx);
      fillCtx.globalAlpha = 1;
    }
    fillCtx.lineWidth = face.subject ? 3 : 2;
    fillCtx.strokeStyle = face.subject ? `rgba(255,255,255,${0.4 + 0.6 * charge})` : 'rgba(255,255,255,0.5)';
    fillCtx.strokeRect(x, y, w, h);
    if (face.reject) {
      const label = document.createElement('span');
      label.textContent = face.reject;
      // the canvas is mirrored, the labels are not: mirror the place
      label.style.left = `${100 - (x + w / 2) / W * 100}%`;
      label.style.top = `${Math.min(96, (y + h) / H * 100 + 1)}%`;
      labels.appendChild(label);
    }
  }
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
