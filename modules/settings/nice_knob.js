// Settings-panel knob: an SVG knob turned by circling the pointer around it. The rotation is
// relative to the grab, so grabbing never jumps the value; the pointer is captured, so circling
// farther out gives finer control.
//
// The knob emits "change" on every new value while the user changes it (drag, wheel, typing); the
// server throttles these. It shows its own value while the user is busy with it; the server answers
// every change, and every external change, with a new "revision", after which the knob shows the
// server's value again.

const SWEEP = 270;          // degrees of travel, the gap at the bottom
const DEAD_ZONE = 8;        // pixels around the center where the pointer's angle is too unstable to use
const FINE = 0.1;           // rotation scale while Shift is held
const BUSY_MS = 200;        // after the last wheel notch, server answers are still older than the knob

function point(angle, radius) {
  const a = (angle * Math.PI) / 180;
  return `${(50 + radius * Math.sin(a)).toFixed(2)} ${(50 - radius * Math.cos(a)).toFixed(2)}`;
}

function arc(from, to, radius) {
  const a0 = Math.min(from, to);
  const a1 = Math.max(from, to);
  if (a1 - a0 < 0.5) return "";
  const large = a1 - a0 > 180 ? 1 : 0;
  return `M ${point(a0, radius)} A ${radius} ${radius} 0 ${large} 1 ${point(a1, radius)}`;
}

export default {
  template: `
    <div class="poser-knob" :class="{ 'poser-knob-readonly': readonly, 'poser-knob-dragging': dragging }">
      <div class="poser-knob-caption">
        <slot></slot>
        <!-- The editor lies over the value, which keeps its place, so typing never resizes the knob. -->
        <div class="poser-knob-value-box">
          <div class="poser-knob-text" :style="{ visibility: editing ? 'hidden' : 'visible' }"
               @click="startEdit">{{ text }}</div>
          <input v-if="editing" ref="editor" class="poser-knob-text poser-knob-editor" :value="text"
                 @keydown.enter.prevent="finishEdit(true)" @keydown.esc.prevent="finishEdit(false)"
                 @blur="finishEdit(true)" />
        </div>
      </div>
      <svg viewBox="0 0 100 100" class="poser-knob-dial"
           @pointerdown="onPointerDown" @pointermove="onPointerMove"
           @pointerup="onPointerUp" @pointercancel="onPointerUp"
           @dblclick="onDoubleClick" @wheel.prevent="onWheel">
        <defs>
          <radialGradient :id="gradientId" cx="40%" cy="35%" r="70%">
            <stop offset="0%" stop-color="#4a4a4a" />
            <stop offset="100%" stop-color="#1c1c1c" />
          </radialGradient>
        </defs>
        <path :d="trackPath" class="poser-knob-track" />
        <path :d="valuePath" class="poser-knob-value" />
        <circle cx="50" cy="50" r="30" :fill="'url(#' + gradientId + ')'" class="poser-knob-body" />
        <line :x1="pointerInner[0]" :y1="pointerInner[1]" :x2="pointerOuter[0]" :y2="pointerOuter[1]"
              class="poser-knob-pointer" />
      </svg>
    </div>`,
  props: {
    value: Number,
    min: Number,
    max: Number,
    step: Number,
    defaultValue: Number,
    decimals: Number,
    readonly: Boolean,
    revision: Number,
  },
  data() {
    return {
      local: null,          // value while the user is changing it; null shows the server's value
      dragging: false,
      dragValue: 0,         // unsnapped drag accumulator
      centerX: 0,
      centerY: 0,
      lastAngle: null,      // pointer angle around the center at the last move; null inside the dead zone
      lastWheel: 0,         // time of the last wheel notch (ms)
      editing: false,
      gradientId: "poser-knob-" + Math.random().toString(36).slice(2),
    };
  },
  computed: {
    shown() {
      return this.local !== null ? this.local : this.value;
    },
    range() {
      return this.max - this.min;
    },
    origin() {
      return this.min < 0 && this.max > 0 ? 0 : this.min;
    },
    text() {
      return this.shown === null || this.shown === undefined ? "" : Number(this.shown).toFixed(this.decimals);
    },
    trackPath() {
      return arc(-SWEEP / 2, SWEEP / 2, 42);
    },
    valuePath() {
      return arc(this.angle(this.origin), this.angle(this.shown), 42);
    },
    pointerInner() {
      return point(this.angle(this.shown), 12).split(" ");
    },
    pointerOuter() {
      return point(this.angle(this.shown), 27).split(" ");
    },
  },
  watch: {
    revision() {
      // The server has answered; show its value unless the user is still changing this knob.
      const wheeling = performance.now() - this.lastWheel < BUSY_MS;
      if (!this.dragging && !wheeling && !this.editing) this.local = null;
    },
  },
  methods: {
    angle(v) {
      if (v === null || v === undefined || !(this.range > 0)) return -SWEEP / 2;
      const f = Math.min(1, Math.max(0, (v - this.min) / this.range));
      return -SWEEP / 2 + SWEEP * f;
    },
    snap(v) {
      const clamped = Math.min(this.max, Math.max(this.min, v));
      const stepped = this.step > 0 ? this.min + Math.round((clamped - this.min) / this.step) * this.step : clamped;
      return Number(Math.min(this.max, Math.max(this.min, stepped)).toFixed(this.decimals));
    },
    change(v) {
      if (v === this.shown) return;
      this.local = v;
      this.$emit("change", v);
    },
    onPointerDown(e) {
      if (this.readonly || e.button !== 0) return;
      e.preventDefault();
      e.currentTarget.setPointerCapture(e.pointerId);
      const rect = e.currentTarget.getBoundingClientRect();
      this.centerX = rect.left + rect.width / 2;
      this.centerY = rect.top + rect.height / 2;
      this.dragging = true;
      this.lastAngle = this.pointerAngle(e);
      this.dragValue = this.shown;
    },
    pointerAngle(e) {
      // Degrees clockwise from 12 o'clock, like the dial; null inside the dead zone.
      const dx = e.clientX - this.centerX;
      const dy = e.clientY - this.centerY;
      if (Math.hypot(dx, dy) < DEAD_ZONE) return null;
      return (Math.atan2(dx, -dy) * 180) / Math.PI;
    },
    onPointerMove(e) {
      if (!this.dragging) return;
      const angle = this.pointerAngle(e);
      if (angle !== null && this.lastAngle !== null) {
        let delta = angle - this.lastAngle;
        if (delta > 180) delta -= 360;
        else if (delta <= -180) delta += 360;
        this.dragValue += (delta / SWEEP) * this.range * (e.shiftKey ? FINE : 1);
        this.dragValue = Math.min(this.max, Math.max(this.min, this.dragValue));
        this.change(this.snap(this.dragValue));
      }
      this.lastAngle = angle;
    },
    onPointerUp(e) {
      if (!this.dragging) return;
      this.dragging = false;
      if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
      // The server's answer to the last change may have come in during the drag.
      if (this.local === this.value) this.local = null;
    },
    onDoubleClick() {
      if (this.readonly || this.defaultValue === null || this.defaultValue === undefined) return;
      this.change(this.snap(this.defaultValue));
    },
    onWheel(e) {
      if (this.readonly) return;
      // Shift+wheel arrives as horizontal scroll in some browsers.
      const delta = e.deltaY || e.deltaX;
      if (!delta) return;
      const increment = e.shiftKey ? this.step : Math.max(this.step, this.range / 100);
      this.lastWheel = performance.now();
      this.change(this.snap(this.shown + (delta < 0 ? increment : -increment)));
    },
    startEdit() {
      if (this.readonly) return;
      this.editing = true;
      this.$nextTick(() => {
        this.$refs.editor.focus();
        this.$refs.editor.select();
      });
    },
    finishEdit(accept) {
      if (!this.editing) return;
      const typed = parseFloat(this.$refs.editor.value);
      this.editing = false;
      if (accept && Number.isFinite(typed)) this.change(this.snap(typed));
    },
  },
};
