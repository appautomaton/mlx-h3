/* mlx-h3 — the packed-sequence instrument.
 *
 * Every number drawn here is derived with the rules the runtime uses, not
 * chosen to look good: 17n+5 frame alignment, f16 spatial and 4x temporal VAE
 * compression, a 1x2x2 patch, 40 Hz stereo audio latents, and the ComfyUI
 * "simple" schedule shifted by 12.0 for video and 3.0 for audio.
 */
(() => {
  'use strict';

  const SVG_NS = 'http://www.w3.org/2000/svg';
  const STEPS = 20;
  const SHIFT_VIDEO = 12.0;
  const SHIFT_AUDIO = 3.0;

  const KINDS = ['text', 'cond', 'audio', 'video'];

  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');

  /* ── model rules ──────────────────────────────────────────────────── */

  /** Frame counts round up to the Video VAE's 17n + 5 alignment. */
  const alignFrames = (f) => {
    const n = Math.max(0, Math.ceil((f - 5) / 17));
    return 17 * n + 5;
  };

  /** Row counts for one packed sequence, drawn with a single first-frame keyframe. */
  function layout(w, h, framesRequested) {
    const frames = alignFrames(framesRequested);

    // f16 spatial compression, then a 2x2 spatial patch: (w/16/2) * (h/16/2).
    const frameRows = (w * h) / 1024;
    // 4x temporal compression.
    const latentT = Math.floor((frames - 1) / 4) + 1;

    // 40 Hz audio latents, two rows per latent for independent L and R.
    const audioT = Math.round((frames / 24) * 40);

    const rows = {
      text: 128,                    // a nominal prompt; text length is the user's
      cond: frameRows,              // one first-frame keyframe
      audio: audioT * 2,
      video: latentT * frameRows,
    };
    rows.total = rows.text + rows.cond + rows.audio + rows.video;
    rows.frames = frames;
    return rows;
  }

  /**
   * ComfyUI BasicScheduler("simple") indexes a 1000-point flow schedule at
   * floor(i * 1000 / steps) and appends a terminal zero, then flow-matching
   * time shift maps it: s*x / (1 + (s - 1)*x).
   */
  function sigmas(steps, shift) {
    const out = [];
    for (let i = 0; i < steps; i++) {
      const base = (1000 - Math.floor((i * 1000) / steps)) / 1000;
      out.push((shift * base) / (1 + (shift - 1) * base));
    }
    out.push(0);
    return out;
  }

  const SIGMA_V = sigmas(STEPS, SHIFT_VIDEO);
  const SIGMA_A = sigmas(STEPS, SHIFT_AUDIO);

  /* ── deterministic noise ──────────────────────────────────────────── */

  /** A small LCG, so the "noise" is identical on every load and every device. */
  function rng(seed) {
    let s = seed >>> 0;
    return () => {
      s = (s * 1664525 + 1013904223) >>> 0;
      return s / 4294967296;
    };
  }

  /* ── elements ─────────────────────────────────────────────────────── */

  const band = document.getElementById('band');
  const ticks = document.getElementById('ticks');
  const curves = document.getElementById('curves');
  if (!band || !curves) return;

  const out = {
    step: document.getElementById('r-step'),
    sv: document.getElementById('r-sv'),
    sa: document.getElementById('r-sa'),
    seq: document.getElementById('r-seq'),
    cost: document.getElementById('r-cost'),
  };

  const BAND_W = 1200;
  const BAND_H = 132;
  const CELL = 12;
  const GAP = 1.6;

  const FILL = {
    text: '#6f6152',
    cond: '#ff7a3d',
    audio: '#63c9b4',
    video: '#ffaa2b',
  };

  let buckets = [];      // buckets[k] = cells revealed at step k
  let current = -1;
  let baseline = null;   // sequence length of the default preset, for relative cost

  /* ── the band ─────────────────────────────────────────────────────── */

  function drawBand(rows) {
    while (band.firstChild) band.removeChild(band.firstChild);
    if (ticks) while (ticks.firstChild) ticks.removeChild(ticks.firstChild);

    buckets = Array.from({ length: STEPS + 1 }, () => []);
    current = -1;

    const rand = rng(0x5eed);
    const frag = document.createDocumentFragment();

    let x = 0;
    KINDS.forEach((kind, ki) => {
      const share = rows[kind] / rows.total;
      const wSeg = share * BAND_W;
      if (wSeg <= 0) { return; }

      const x0 = x;
      const x1 = x + wSeg;
      x = x1;

      // A segment boundary tick, so the true proportions stay legible even
      // where a segment is only a few pixels wide.
      if (ki > 0 && ticks) {
        const t = document.createElement('span');
        t.className = 'band__tick';
        t.style.left = `${(x0 / BAND_W) * 100}%`;
        ticks.appendChild(t);
      }

      for (let cx = x0; cx < x1 - 0.5; cx += CELL) {
        const cw = Math.min(CELL - GAP, x1 - cx - GAP);
        if (cw <= 0.2) continue;
        for (let cy = 0; cy < BAND_H; cy += CELL) {
          const r = document.createElementNS(SVG_NS, 'rect');
          r.setAttribute('x', cx.toFixed(2));
          r.setAttribute('y', cy.toFixed(2));
          r.setAttribute('width', cw.toFixed(2));
          r.setAttribute('height', (CELL - GAP).toFixed(2));
          r.setAttribute('fill', FILL[kind]);
          // An uneven floor, so the unresolved band reads as a field of noise
          // rather than an empty box.
          const floor = (0.05 + rand() * 0.13).toFixed(3);
          r.dataset.floor = floor;
          // Resolved cells vary too, so a settled segment keeps some grain
          // instead of reading as one flat block of colour.
          r.dataset.peak = (0.76 + rand() * 0.24).toFixed(3);
          r.setAttribute('opacity', floor);
          r.style.transition = 'opacity .5s cubic-bezier(.2,.7,.2,1)';

          // Each cell resolves at its own step, so the band fills in the way a
          // denoiser does: unevenly, then all at once near the end.
          const k = Math.max(1, Math.ceil(rand() * STEPS));
          buckets[k].push(r);
          frag.appendChild(r);
        }
      }
    });

    band.appendChild(frag);
  }

  function setBandStep(step) {
    if (step === current) return;
    if (step < current) {
      // rewind
      for (let k = 1; k <= STEPS; k++) {
        for (const r of buckets[k]) r.setAttribute('opacity', r.dataset.floor);
      }
      current = 0;
    }
    for (let k = current + 1; k <= step; k++) {
      if (!buckets[k]) continue;
      for (const r of buckets[k]) r.setAttribute('opacity', r.dataset.peak);
    }
    current = step;
  }

  /* ── the sigma curves ─────────────────────────────────────────────── */

  /* The curves are drawn in real CSS pixels rather than a fixed viewBox, so the
     axis labels render at exactly their stated size instead of shrinking to
     nothing on a phone. */
  const C = { w: 1200, h: 190, l: 56, r: 16, t: 14, b: 34 };

  function measure() {
    const box = curves.getBoundingClientRect();
    C.w = Math.max(280, Math.round(box.width) || 1200);
    C.h = Math.max(120, Math.round(box.height) || 190);
    C.l = C.w < 520 ? 38 : 56;
    C.r = C.w < 520 ? 10 : 16;
    C.b = C.w < 520 ? 30 : 34;
    curves.setAttribute('viewBox', `0 0 ${C.w} ${C.h}`);
  }

  const px = (i) => C.l + (i / STEPS) * (C.w - C.l - C.r);
  const py = (s) => C.t + (1 - s) * (C.h - C.t - C.b);

  const el = (name, attrs, cls) => {
    const n = document.createElementNS(SVG_NS, name);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (cls) n.setAttribute('class', cls);
    return n;
  };

  let dotV, dotA, scan;

  function drawCurves() {
    measure();
    while (curves.firstChild) curves.removeChild(curves.firstChild);
    const g = document.createDocumentFragment();
    const tight = C.w < 520;

    // horizontal grid at sigma 0, .5, 1
    [0, 0.5, 1].forEach((s) => {
      g.appendChild(el('line', { x1: C.l, y1: py(s), x2: C.w - C.r, y2: py(s) }, 'cv-grid'));
      g.appendChild(el('text', { x: C.l - 8, y: py(s) + 4, 'text-anchor': 'end' }, 'cv-label'))
        .textContent = s.toFixed(1);
    });

    // step ticks
    for (let i = 0; i <= STEPS; i += tight ? 10 : 5) {
      g.appendChild(el('line', { x1: px(i), y1: py(0), x2: px(i), y2: py(0) + 5 }, 'cv-axis'));
      g.appendChild(el('text', { x: px(i), y: py(0) + 20, 'text-anchor': 'middle' }, 'cv-label'))
        .textContent = String(i);
    }
    const path = (sig) =>
      sig.map((s, i) => `${i ? 'L' : 'M'}${px(i).toFixed(1)},${py(s).toFixed(1)}`).join(' ');
    const area = (sig) =>
      `${path(sig)} L${px(STEPS).toFixed(1)},${py(0).toFixed(1)} L${px(0).toFixed(1)},${py(0).toFixed(1)} Z`;

    g.appendChild(el('path', { d: area(SIGMA_V) }, 'cv-vid-a'));
    g.appendChild(el('path', { d: area(SIGMA_A) }, 'cv-aud-a'));

    scan = el('line', { x1: px(0), y1: C.t, x2: px(0), y2: py(0) }, 'cv-scan');
    g.appendChild(scan);

    g.appendChild(el('path', { d: path(SIGMA_A) }, 'cv-aud'));
    g.appendChild(el('path', { d: path(SIGMA_V) }, 'cv-vid'));

    // Labels sit on the curves themselves, so neither needs a separate key.
    g.appendChild(el('text', { x: px(13.4), y: py(SIGMA_V[13]) - 12 }, 'cv-tag cv-tag--v'))
      .textContent = tight ? 'video 12.0' : 'video · shift 12.0';
    g.appendChild(el('text', { x: px(5.4), y: py(SIGMA_A[5]) - 12 }, 'cv-tag cv-tag--a'))
      .textContent = tight ? 'audio 3.0' : 'audio · shift 3.0';

    dotA = el('circle', { cx: px(0), cy: py(SIGMA_A[0]), r: 4.5 }, 'cv-dot-a');
    dotV = el('circle', { cx: px(0), cy: py(SIGMA_V[0]), r: 4.5 }, 'cv-dot-v');
    g.appendChild(dotA);
    g.appendChild(dotV);

    curves.appendChild(g);
  }

  function setCurveStep(step) {
    const x = px(step);
    if (scan) { scan.setAttribute('x1', x); scan.setAttribute('x2', x); }
    if (dotV) { dotV.setAttribute('cx', x); dotV.setAttribute('cy', py(SIGMA_V[step])); }
    if (dotA) { dotA.setAttribute('cx', x); dotA.setAttribute('cy', py(SIGMA_A[step])); }
  }

  /* ── readouts ─────────────────────────────────────────────────────── */

  const fmt = (n) => n.toLocaleString('en-US');

  function setRows(rows) {
    for (const kind of KINDS) {
      const cell = document.querySelector(`.legend__v[data-rows="${kind}"]`);
      if (cell) cell.textContent = fmt(rows[kind]);
    }
    if (out.seq) out.seq.textContent = fmt(rows.total);
    if (out.cost) {
      const ratio = (rows.total / baseline) ** 2;
      out.cost.textContent = `${ratio < 10 ? ratio.toFixed(2) : ratio.toFixed(1)}×`;
    }
  }

  let stepNow = 0;

  function setStep(step) {
    stepNow = step;
    setBandStep(step);
    setCurveStep(step);
    if (out.step) out.step.textContent = `${String(step).padStart(2, '0')} / ${STEPS}`;
    if (out.sv) out.sv.textContent = SIGMA_V[step].toFixed(4);
    if (out.sa) out.sa.textContent = SIGMA_A[step].toFixed(4);
  }

  /* ── the loop ─────────────────────────────────────────────────────── */

  let timer = null;
  let running = false;

  function stop() {
    running = false;
    if (timer) { clearTimeout(timer); timer = null; }
  }

  function play(from = 0, lead = 700) {
    stop();
    running = true;
    let i = from;
    setStep(i);

    const tick = () => {
      if (!running) return;
      i += 1;
      if (i > STEPS) {
        // hold the resolved sequence, then start over
        timer = setTimeout(() => { if (running) play(0, 260); }, 2200);
        return;
      }
      setStep(i);
      timer = setTimeout(tick, i === 1 ? 420 : 190);
    };
    // A shorter lead on replay, so the reset does not sit on an empty band.
    timer = setTimeout(tick, lead);
  }

  /* ── build ────────────────────────────────────────────────────────── */

  function build(w, h, frames, animate) {
    const rows = layout(w, h, frames);
    if (baseline === null) baseline = rows.total;
    drawBand(rows);
    setRows(rows);

    if (animate && !reduced.matches) {
      play(0);
    } else {
      stop();
      setStep(STEPS);
    }
  }

  drawCurves();

  const presets = Array.from(document.querySelectorAll('.preset'));
  const active = () => presets.find((b) => b.classList.contains('is-on')) || presets[0];

  const read = (b) => [Number(b.dataset.w), Number(b.dataset.h), Number(b.dataset.f)];

  // The default preset defines the baseline for relative attention cost, so it
  // must be measured before anything else is drawn.
  {
    const [w, h, f] = read(active());
    baseline = layout(w, h, f).total;
  }

  presets.forEach((btn) => {
    btn.addEventListener('click', () => {
      presets.forEach((b) => b.classList.toggle('is-on', b === btn));
      const [w, h, f] = read(btn);
      build(w, h, f, true);
    });
  });

  {
    // Start at step zero when motion is allowed, so the first thing on screen is
    // the noise the run begins from rather than a resolved band that then resets.
    const [w, h, f] = read(active());
    build(w, h, f, false);
    if (!reduced.matches) setStep(0);
  }

  // The curves are drawn in pixel space, so a width change means a redraw.
  let resizeTimer = null;
  let lastWidth = Math.round(curves.getBoundingClientRect().width);
  window.addEventListener('resize', () => {
    if (resizeTimer) clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      const w = Math.round(curves.getBoundingClientRect().width);
      if (w === lastWidth) return;
      lastWidth = w;
      drawCurves();
      setCurveStep(stepNow);
    }, 150);
  });

  // Only run the loop while the instrument is on screen.
  const figure = document.getElementById('instrument');
  if (figure && 'IntersectionObserver' in window) {
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting && !reduced.matches) {
            if (!running) play(0);
          } else {
            stop();
          }
        }
      },
      { threshold: 0.25 }
    );
    io.observe(figure);
  } else if (!reduced.matches) {
    play(0);
  }

  reduced.addEventListener('change', () => {
    if (reduced.matches) { stop(); setStep(STEPS); } else { play(0); }
  });

  /* ── copy buttons ─────────────────────────────────────────────────── */

  document.querySelectorAll('.copy').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const target = document.querySelector(btn.dataset.copy);
      if (!target) return;
      const text = target.textContent.trim();
      try {
        await navigator.clipboard.writeText(text);
      } catch {
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.setAttribute('readonly', '');
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); } catch { /* nothing else to try */ }
        document.body.removeChild(ta);
      }
      const was = btn.textContent;
      btn.textContent = 'copied';
      btn.classList.add('is-done');
      setTimeout(() => { btn.textContent = was; btn.classList.remove('is-done'); }, 1600);
    });
  });
})();
