/**
 * Medical-Grade HTML5 Canvas Oscilloscope Engine
 * High-performance, multi-channel biomedical waveform renderer
 * Supports real-time phosphor glow, continuous sweep & scrolling modes,
 * clinical grid graticules, R-peak / systolic annotations, and sensor fault overlays.
 */

class MedicalOscilloscope {
  /**
   * @param {HTMLCanvasElement} canvas - Target canvas element
   * @param {Object} options - Configuration options
   */
  constructor(canvas, options = {}) {
    if (!canvas) throw new Error("Canvas element is required for MedicalOscilloscope");
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d", { alpha: false });

    // Configuration defaults
    this.mode = options.mode || "sweep"; // 'sweep' (clinical CRT sweep) or 'scroll' (strip-chart)
    this.sampleRate = options.sampleRate || 100; // Hz
    this.sweepDuration = options.sweepDuration || 5.0; // Seconds to sweep canvas width
    this.gridColor = options.gridColor || "rgba(255, 255, 255, 0.04)";
    this.gridMajorColor = options.gridMajorColor || "rgba(255, 255, 255, 0.08)";
    this.backgroundColor = options.backgroundColor || "#0b0f17";
    this.showGrid = options.showGrid !== undefined ? options.showGrid : true;
    this.showRuler = options.showRuler !== undefined ? options.showRuler : true;
    this.glowIntensity = options.glowIntensity !== undefined ? options.glowIntensity : 0; // Apple Watch ECG: crisp line without neon bloom
    this.title = options.title || "WAVEFORM MONITOR";
    this.units = options.units || "mV";

    // Channels configuration
    // Each channel: { id, name, color, glowColor, gain, offset, minVal, maxVal, visible, dataBuffer }
    this.channels = [];
    if (options.channels && options.channels.length > 0) {
      options.channels.forEach((ch) => this.addChannel(ch));
    } else {
      this.addChannel({
        id: "ch1",
        name: "LEAD II",
        color: "#15803d",
        glowColor: "transparent",
        gain: 1.0,
      });
    }

    // Oscilloscope State
    this.isFrozen = false;
    this.isLeadsOff = false;
    this.leadsOffMessage = "LEADS OFF - REATTACH ELECTRODES";
    this.sweepX = 0; // Current horizontal draw head for sweep mode
    this.prevSweepX = 0;
    this.sweepHeadWidth = 24; // Eraser gap ahead of sweep head

    // Sample buffers for rendering
    // For sweep mode: array of samples mapping to x pixel columns
    this.totalSamplesInSweep = Math.round(this.sampleRate * this.sweepDuration);
    this.sweepIndex = 0;

    // Measurement cursors
    this.enableCursor = true;
    this.cursorPos = null; // {x, y}
    this._initEvents();

    // High DPI Canvas Scaling
    this.dpr = window.devicePixelRatio || 1;
    this.resize();

    // Internal animation loop controller
    this.rafId = null;
    this.lastRenderTime = performance.now();
    this.fps = 60;
  }

  /**
   * Add a channel to the oscilloscope
   */
  addChannel(config) {
    const ch = {
      id: config.id || `ch_${this.channels.length + 1}`,
      name: config.name || `CHANNEL ${this.channels.length + 1}`,
      color: config.color || "#15803d",
      glowColor: config.glowColor || "transparent",
      gain: config.gain || 1.0,
      offsetY: config.offsetY || 0, // baseline offset relative to center (-1 to 1)
      visible: config.visible !== undefined ? config.visible : true,
      lineWidth: config.lineWidth || 2.0,
      minVal: config.minVal !== undefined ? config.minVal : -1.5,
      maxVal: config.maxVal !== undefined ? config.maxVal : 2.5,
      autoScale: config.autoScale !== undefined ? config.autoScale : true,
      samples: [], // Ring buffer for scroll mode
      sweepPoints: [], // Points for sweep mode: [{x, y, isPeak, label}]
      lastSampleY: null,
      peakMarkers: [], // Detected peaks [{x, y, label, time}]
    };
    this.channels.push(ch);
    return ch;
  }

  /**
   * Resize canvas to container with High DPI Retina support
   */
  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const width = Math.max(100, Math.floor(rect.width));
    const height = Math.max(80, Math.floor(rect.height));

    this.dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.floor(width * this.dpr);
    this.canvas.height = Math.floor(height * this.dpr);

    this.width = width;
    this.height = height;

    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this.renderStaticBackground();
  }

  /**
   * Push a batch of incoming samples into channels
   * @param {Object} sampleBatch - { chId: { value, isPeak, label }, ... } or array
   */
  pushData(sampleBatch) {
    if (this.isFrozen) return;

    if (Array.isArray(sampleBatch)) {
      sampleBatch.forEach((s) => this._processSingleSample(s));
    } else {
      this._processSingleSample(sampleBatch);
    }
  }

  _processSingleSample(sample) {
    this.channels.forEach((ch) => {
      let val = 0;
      let isPeak = false;
      let peakLabel = "";

      if (typeof sample === "number") {
        val = sample;
      } else if (sample && sample[ch.id] !== undefined) {
        const item = sample[ch.id];
        if (typeof item === "number") {
          val = item;
        } else if (item && typeof item === "object") {
          val = item.value !== undefined ? item.value : 0;
          isPeak = !!item.isPeak;
          peakLabel = item.label || "";
        }
      } else if (sample && sample.value !== undefined && this.channels.length === 1) {
        val = sample.value;
        isPeak = !!sample.isPeak;
        peakLabel = sample.label || "";
      }

      // If leads off, add jitter flatline
      if (this.isLeadsOff) {
        val = (Math.random() - 0.5) * 0.05;
      }

      // Normalize value to screen coordinate Y
      const y = this._valueToY(val, ch);

      if (this.mode === "sweep") {
        const x = (this.sweepIndex / this.totalSamplesInSweep) * this.width;
        ch.sweepPoints[this.sweepIndex] = { x, y, isPeak, peakLabel, val };

        if (isPeak) {
          ch.peakMarkers.push({ x, y, label: peakLabel, ttl: 120 });
        }
      } else {
        // Scroll mode
        ch.samples.push({ y, isPeak, peakLabel, val });
        if (ch.samples.length > this.width) {
          ch.samples.shift();
        }
      }
    });

    if (this.mode === "sweep") {
      this.prevSweepX = (this.sweepIndex / this.totalSamplesInSweep) * this.width;
      this.sweepIndex++;
      if (this.sweepIndex >= this.totalSamplesInSweep) {
        this.sweepIndex = 0;
      }
      this.sweepX = (this.sweepIndex / this.totalSamplesInSweep) * this.width;
    }
  }

  _valueToY(val, ch) {
    const range = ch.maxVal - ch.minVal || 1.0;
    const normalized = (val - ch.minVal) / range; // 0 to 1
    const center = this.height * 0.5;
    const heightSpan = this.height * 0.85;

    // Apply gain & baseline offset
    let y = center - (normalized - 0.5) * heightSpan * ch.gain + ch.offsetY * (this.height * 0.3);
    // Clamp to canvas padding
    return Math.max(12, Math.min(this.height - 12, y));
  }

  _yToValue(y, ch) {
    const center = this.height * 0.5;
    const heightSpan = this.height * 0.85;
    const range = ch.maxVal - ch.minVal || 1.0;
    const normalized = 0.5 - (y - center - ch.offsetY * (this.height * 0.3)) / (heightSpan * ch.gain);
    return ch.minVal + normalized * range;
  }

  /**
   * Main render method called per animation frame
   */
  render() {
    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;

    if (!w || !h) return;

    // 1. Clear background
    ctx.fillStyle = this.backgroundColor;
    ctx.fillRect(0, 0, w, h);

    // 2. Clinical Graticule Grid
    if (this.showGrid) {
      this._drawMedicalGrid(ctx, w, h);
    }

    // 3. Calibration pulse in top-left
    if (this.showRuler) {
      this._drawCalibrationPulse(ctx);
    }

    // 4. Render Waveforms per channel
    this.channels.forEach((ch) => {
      if (!ch.visible) return;

      if (this.mode === "sweep") {
        this._renderSweepChannel(ctx, ch, w, h);
      } else {
        this._renderScrollChannel(ctx, ch, w, h);
      }
    });

    // 5. Draw sweep head bar and erase front curtain in sweep mode
    if (this.mode === "sweep" && !this.isFrozen) {
      this._drawSweepHead(ctx, w, h);
    }

    // 6. Draw alerts overlay (e.g. LEADS OFF / FINGER OFF)
    if (this.isLeadsOff) {
      this._drawAlertOverlay(ctx, w, h, this.leadsOffMessage);
    }

    // 7. Measurement Cursor HUD
    if (this.enableCursor && this.cursorPos) {
      this._drawCursorOverlay(ctx, w, h);
    }

    // 8. Header / Info overlay
    this._drawHUDLabels(ctx, w, h);
  }

  _drawMedicalGrid(ctx, w, h) {
    // Standard ECG paper: 5mm major squares, 1mm minor squares
    // 25mm = 1 sec, 10mm = 1 mV.
    const minorStep = 10;
    const majorStep = 50;

    ctx.save();
    ctx.lineWidth = 1;

    // Minor dot grid or subtle lines
    ctx.strokeStyle = this.gridColor;
    ctx.beginPath();
    for (let x = 0; x <= w; x += minorStep) {
      if (x % majorStep !== 0) {
        ctx.moveTo(x + 0.5, 0);
        ctx.lineTo(x + 0.5, h);
      }
    }
    for (let y = 0; y <= h; y += minorStep) {
      if (y % majorStep !== 0) {
        ctx.moveTo(0, y + 0.5);
        ctx.lineTo(w, y + 0.5);
      }
    }
    ctx.stroke();

    // Major grid lines
    ctx.strokeStyle = this.gridMajorColor;
    ctx.beginPath();
    for (let x = 0; x <= w; x += majorStep) {
      ctx.moveTo(x + 0.5, 0);
      ctx.lineTo(x + 0.5, h);
    }
    for (let y = 0; y <= h; y += majorStep) {
      ctx.moveTo(0, y + 0.5);
      ctx.lineTo(w, y + 0.5);
    }
    ctx.stroke();

    ctx.restore();
  }

  _drawCalibrationPulse(ctx) {
    // 1mV reference step at the beginning of the trace (50px wide x 30px high)
    ctx.save();
    ctx.strokeStyle = "rgba(148, 163, 184, 0.4)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([2, 2]);

    const baseY = this.height * 0.85;
    ctx.beginPath();
    ctx.moveTo(8, baseY);
    ctx.lineTo(16, baseY);
    ctx.lineTo(16, baseY - 25);
    ctx.lineTo(28, baseY - 25);
    ctx.lineTo(28, baseY);
    ctx.lineTo(36, baseY);
    ctx.stroke();

    ctx.fillStyle = "rgba(148, 163, 184, 0.6)";
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.fillText("1 mV", 14, baseY - 28);
    ctx.restore();
  }

  _renderSweepChannel(ctx, ch, w, h) {
    const points = ch.sweepPoints;
    if (!points || points.length === 0) return;

    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    // Apple Watch ECG Style: Crisp, clean stroke with zero neon bloom/blur
    ctx.shadowBlur = 0;
    ctx.shadowColor = "transparent";
    ctx.strokeStyle = ch.color;
    ctx.lineWidth = ch.lineWidth || 2.0;

    // Segment 1: from sweepX + headWidth to end of screen (past sweep)
    // Segment 2: from 0 to sweepX (fresh trace)
    const currentIdx = this.sweepIndex;
    const total = this.totalSamplesInSweep;
    const eraseMargin = Math.ceil((this.sweepHeadWidth / w) * total);

    // Draw old trace (with slight dimmer opacity for clean persistence)
    ctx.globalAlpha = 0.55;
    ctx.beginPath();
    let started = false;
    const startOld = (currentIdx + eraseMargin) % total;

    for (let i = startOld; i < total; i++) {
      const pt = points[i];
      if (!pt) continue;
      if (!started) {
        ctx.moveTo(pt.x, pt.y);
        started = true;
      } else {
        ctx.lineTo(pt.x, pt.y);
      }
    }
    ctx.stroke();

    // Draw fresh trace (bright full intensity)
    ctx.globalAlpha = 1.0;
    ctx.beginPath();
    started = false;
    for (let i = 0; i < currentIdx; i++) {
      const pt = points[i];
      if (!pt) continue;
      if (!started) {
        ctx.moveTo(pt.x, pt.y);
        started = true;
      } else {
        ctx.lineTo(pt.x, pt.y);
      }
    }
    ctx.stroke();

    // Render Peak Markers (e.g. R-peaks or Systolic peaks)
    if (ch.peakMarkers && ch.peakMarkers.length > 0) {
      this._renderPeaks(ctx, ch);
    }

    ctx.restore();
  }

  _renderScrollChannel(ctx, ch, w, h) {
    const samples = ch.samples;
    if (samples.length < 2) return;

    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    // Apple Watch ECG Style: Crisp, clean stroke with zero neon bloom/blur
    ctx.shadowBlur = 0;
    ctx.shadowColor = "transparent";
    ctx.strokeStyle = ch.color;
    ctx.lineWidth = ch.lineWidth || 2.0;

    const step = w / (this.totalSamplesInSweep - 1);
    const startX = Math.max(0, w - samples.length * step);

    ctx.beginPath();
    samples.forEach((s, idx) => {
      const x = startX + idx * step;
      if (idx === 0) {
        ctx.moveTo(x, s.y);
      } else {
        ctx.lineTo(x, s.y);
      }

      if (s.isPeak) {
        this._drawPeakPoint(ctx, x, s.y, s.peakLabel || "R", ch.color);
      }
    });
    ctx.stroke();
    ctx.restore();
  }

  _drawSweepHead(ctx, w, h) {
    // Dark clearing rectangle ahead of the sweep head
    ctx.save();
    ctx.fillStyle = this.backgroundColor;
    const eraseX = this.sweepX;
    ctx.fillRect(eraseX, 0, this.sweepHeadWidth, h);

    // Subtle Apple-style vertical cursor line
    const gradient = ctx.createLinearGradient(eraseX, 0, eraseX + 4, 0);
    gradient.addColorStop(0, "rgba(21, 128, 61, 0.75)");
    gradient.addColorStop(0.3, "rgba(21, 128, 61, 0.25)");
    gradient.addColorStop(1, "rgba(21, 128, 61, 0)");

    ctx.fillStyle = gradient;
    ctx.fillRect(eraseX - 2, 0, 8, h);
    ctx.restore();
  }

  _renderPeaks(ctx, ch) {
    for (let i = ch.peakMarkers.length - 1; i >= 0; i--) {
      const p = ch.peakMarkers[i];
      p.ttl -= 1;
      if (p.ttl <= 0) {
        ch.peakMarkers.splice(i, 1);
        continue;
      }
      this._drawPeakPoint(ctx, p.x, p.y, p.label || "R", ch.color);
    }
  }

  _drawPeakPoint(ctx, x, y, label, color) {
    ctx.save();
    ctx.fillStyle = color;
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;

    // Diamond marker
    ctx.beginPath();
    ctx.moveTo(x, y - 7);
    ctx.lineTo(x + 5, y);
    ctx.lineTo(x, y + 7);
    ctx.lineTo(x - 5, y);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Peak label (e.g. 'R' or HR number)
    ctx.fillStyle = "#ffffff";
    ctx.font = '600 10px "SF Mono", "JetBrains Mono", monospace';
    ctx.textAlign = "center";
    ctx.fillText(label, x, y - 11);
    ctx.restore();
  }

  _drawAlertOverlay(ctx, w, h, message) {
    ctx.save();
    // Translucent alert background
    ctx.fillStyle = "rgba(190, 18, 60, 0.08)";
    ctx.fillRect(0, 0, w, h);

    // Pulsing central alert badge
    const pulse = (Math.sin(performance.now() * 0.006) + 1) * 0.5;
    ctx.strokeStyle = `rgba(190, 18, 60, ${0.4 + pulse * 0.4})`;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(w * 0.15, h * 0.35, w * 0.7, h * 0.3);

    ctx.fillStyle = `rgba(17, 24, 39, 0.92)`;
    ctx.fillRect(w * 0.15 + 1, h * 0.35 + 1, w * 0.7 - 2, h * 0.3 - 2);

    ctx.fillStyle = "#be123c";
    ctx.font = '600 13px -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", sans-serif';
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(`⚠️ ${message}`, w * 0.5, h * 0.5);

    ctx.restore();
  }

  _drawCursorOverlay(ctx, w, h) {
    const pos = this.cursorPos;
    if (pos.x < 0 || pos.x > w || pos.y < 0 || pos.y > h) return;

    ctx.save();
    ctx.strokeStyle = "rgba(248, 250, 252, 0.35)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);

    // Crosshairs
    ctx.beginPath();
    ctx.moveTo(pos.x, 0);
    ctx.lineTo(pos.x, h);
    ctx.moveTo(0, pos.y);
    ctx.lineTo(w, pos.y);
    ctx.stroke();

    // Measurement badge
    const ch = this.channels[0];
    const val = ch ? this._yToValue(pos.y, ch).toFixed(2) : 0;
    const timeSec = ((pos.x / w) * this.sweepDuration).toFixed(2);
    const text = `T: ${timeSec}s | V: ${val} ${this.units}`;

    ctx.setLineDash([]);
    ctx.font = '10px "SF Mono", "JetBrains Mono", monospace';
    const textWidth = ctx.measureText(text).width + 12;

    const badgeX = Math.min(w - textWidth - 6, Math.max(6, pos.x + 8));
    const badgeY = Math.max(20, Math.min(h - 24, pos.y - 8));

    ctx.fillStyle = "rgba(17, 24, 39, 0.92)";
    ctx.strokeStyle = "rgba(29, 78, 216, 0.5)";
    ctx.fillRect(badgeX, badgeY - 14, textWidth, 20);
    ctx.strokeRect(badgeX, badgeY - 14, textWidth, 20);

    ctx.fillStyle = "#1d4ed8";
    ctx.textAlign = "left";
    ctx.fillText(text, badgeX + 6, badgeY);

    ctx.restore();
  }

  _drawHUDLabels(ctx, w, h) {
    ctx.save();
    ctx.font = '10px "SF Mono", "JetBrains Mono", monospace';

    // Channel badges in top right
    let rightOffset = w - 12;
    for (let i = this.channels.length - 1; i >= 0; i--) {
      const ch = this.channels[i];
      if (!ch.visible) continue;

      const label = `${ch.name} [x${ch.gain}]`;
      const width = ctx.measureText(label).width + 16;
      rightOffset -= width;

      ctx.fillStyle = "rgba(17, 24, 39, 0.85)";
      ctx.strokeStyle = ch.color;
      ctx.lineWidth = 1;
      ctx.fillRect(rightOffset, 8, width, 18);
      ctx.strokeRect(rightOffset, 8, width, 18);

      ctx.fillStyle = ch.color;
      ctx.textAlign = "left";
      ctx.fillText(label, rightOffset + 8, 21);

      rightOffset -= 8;
    }

    // Title in top left
    ctx.fillStyle = "rgba(248, 250, 252, 0.85)";
    ctx.font = '600 11px -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", sans-serif';
    ctx.textAlign = "left";
    ctx.fillText(this.title, 12, 20);

    // Speed / Scale tag
    ctx.fillStyle = "rgba(148, 163, 184, 0.6)";
    ctx.font = '9px "SF Mono", "JetBrains Mono", monospace';
    ctx.fillText(`SWEEP: ${this.sweepDuration}s (${Math.round(w / this.sweepDuration)} px/s)`, 12, h - 8);

    ctx.restore();
  }

  _initEvents() {
    this.canvas.addEventListener("mousemove", (e) => {
      const rect = this.canvas.getBoundingClientRect();
      this.cursorPos = {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
      };
    });

    this.canvas.addEventListener("mouseleave", () => {
      this.cursorPos = null;
    });

    // Window resize observer
    if (window.ResizeObserver) {
      const ro = new ResizeObserver(() => this.resize());
      ro.observe(this.canvas);
    } else {
      window.addEventListener("resize", () => this.resize());
    }
  }

  // Public control methods
  setGain(chId, gain) {
    const ch = this.channels.find((c) => c.id === chId) || this.channels[0];
    if (ch) ch.gain = gain;
  }

  setSweepDuration(seconds) {
    this.sweepDuration = Math.max(1.0, Math.min(20.0, seconds));
    this.totalSamplesInSweep = Math.round(this.sampleRate * this.sweepDuration);
    this.sweepIndex = 0;
    this.channels.forEach((ch) => {
      ch.sweepPoints = [];
    });
  }

  setMode(mode) {
    if (mode === "sweep" || mode === "scroll") {
      this.mode = mode;
      this.sweepIndex = 0;
      this.channels.forEach((ch) => {
        ch.sweepPoints = [];
        ch.samples = [];
      });
    }
  }

  setLeadsOff(isOff, message) {
    this.isLeadsOff = isOff;
    if (message) this.leadsOffMessage = message;
  }

  toggleFreeze() {
    this.isFrozen = !this.isFrozen;
    return this.isFrozen;
  }

  reset() {
    this.sweepIndex = 0;
    this.channels.forEach((ch) => {
      ch.sweepPoints = [];
      ch.samples = [];
      ch.peakMarkers = [];
    });
  }

  renderStaticBackground() {
    this.render();
  }
}

// Export for ES module or browser window
if (typeof window !== "undefined") {
  window.MedicalOscilloscope = MedicalOscilloscope;
}
if (typeof module !== "undefined" && module.exports) {
  module.exports = { MedicalOscilloscope };
}
