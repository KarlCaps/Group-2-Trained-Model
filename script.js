// ---------- CNN forward pass in plain JavaScript (no libraries, runs offline) ----------
// Layers come from weights.js (exported from the trained Keras model).

function conv2d(x, H, W, C, layer) {
    const [kh, kw, , F] = layer.shape;
    const oh = H - kh + 1, ow = W - kw + 1;
  const out = new Float32Array(oh * ow * F);
    const w = layer.w, b = layer.b;
    for (let i = 0; i < oh; i++) {
    for (let j = 0; j < ow; j++) {
        for (let f = 0; f < F; f++) {
        let s = b[f];
        for (let di = 0; di < kh; di++) {
            for (let dj = 0; dj < kw; dj++) {
            const xi = ((i + di) * W + (j + dj)) * C;
            const wi = ((di * kw + dj) * C) * F + f;
            for (let c = 0; c < C; c++) s += x[xi + c] * w[wi + c * F];
            }
        }
        out[(i * ow + j) * F + f] = s > 0 ? s : 0; // ReLU
        }
    }
    }
    return { data: out, H: oh, W: ow, C: F };
}

function maxPool(t) {
    const oh = t.H >> 1, ow = t.W >> 1, C = t.C;
  const out = new Float32Array(oh * ow * C);
    for (let i = 0; i < oh; i++)
    for (let j = 0; j < ow; j++)
        for (let c = 0; c < C; c++) {
        let m = -Infinity;
        for (let di = 0; di < 2; di++)
            for (let dj = 0; dj < 2; dj++) {
            const v = t.data[((2 * i + di) * t.W + (2 * j + dj)) * C + c];
            if (v > m) m = v;
            }
        out[(i * ow + j) * C + c] = m;
        }
    return { data: out, H: oh, W: ow, C };
}

function dense(x, layer, relu) {
    const [n, m] = layer.shape;
    const out = new Float32Array(m);
    for (let k = 0; k < m; k++) out[k] = layer.b[k];
    for (let i = 0; i < n; i++) {
    const xi = x[i];
    if (xi === 0) continue;
    const base = i * m;
    for (let k = 0; k < m; k++) out[k] += xi * layer.w[base + k];
    }
    if (relu) for (let k = 0; k < m; k++) if (out[k] < 0) out[k] = 0;
    return out;
}

function softmax(z) {
    const mx = Math.max(...z);
    const e = Array.from(z, v => Math.exp(v - mx));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map(v => v / s);
}

// image: Float32Array(784) with values 0..1 (white digit on black)
function predictCNN(image) {
  const L = MODEL.layers; // conv, conv, conv, conv, dense, dense
    let t = { data: image, H: 28, W: 28, C: 1 };
    t = conv2d(t.data, t.H, t.W, t.C, L[0]);
    t = conv2d(t.data, t.H, t.W, t.C, L[1]);
    t = maxPool(t);
    t = conv2d(t.data, t.H, t.W, t.C, L[2]);
    t = conv2d(t.data, t.H, t.W, t.C, L[3]);
    t = maxPool(t);
    let v = dense(t.data, L[4], true);
    v = dense(v, L[5], false);
    return softmax(v);
}

// ---------- UI ----------
(function () {
    if (typeof document === "undefined") return;
    const $ = id => document.getElementById(id);

    const pad = $("pad"), ctx = pad.getContext("2d");
    const seen = $("seen"), sctx = seen.getContext("2d");
    const hint = $("hint"), status = $("status"), msg = $("msg");
    const digitEl = $("digit"), confEl = $("conf"), bars = $("bars"), auto = $("auto");

  const MIN_SIZE = 45;        // drawing smaller than this (in a 280px pad) is treated as a dot / too small
  const SURE = 70;            // below this confidence (%) the model says it is not sure

  // ----- fill the stats from the exported model (meta) -----
    const meta = (typeof MODEL !== "undefined" && MODEL.meta) ? MODEL.meta : null;
    if (meta) {
        const nf = n => n.toLocaleString("en-US");
        $("chipAcc").textContent = meta.accuracy.toFixed(2) + "%";
        $("chipCorrect").textContent = nf(meta.correct) + " / " + nf(meta.total);
        $("chipEpochs").textContent = meta.epochs + " epochs";
        $("howEpochs").textContent = meta.epochs;
        $("howResult").textContent = nf(meta.correct) + " correct, " + nf(meta.total - meta.correct) + " wrong";
    }

  // ----- probability bars 0..9 -----
    for (let i = 0; i < 10; i++) {
    bars.insertAdjacentHTML("beforeend",
        `<div class="bar" id="bar${i}"><span>${i}</span><div class="track"><div class="fill"></div></div><span class="pc">0%</span></div>`);
    }
    function resetResult() {
    digitEl.textContent = "?"; digitEl.classList.remove("unsure");
    confEl.textContent = "0";
    document.querySelectorAll(".bar").forEach(b => {
        b.classList.remove("top"); b.querySelector(".fill").style.width = "0"; b.querySelector(".pc").textContent = "0%";
    });
    }
    function say(text, warn) { msg.textContent = text; msg.classList.toggle("warn", !!warn); }

  // ----- drawing pad -----
    function resetPad() {
    ctx.fillStyle = "#000"; ctx.fillRect(0, 0, 280, 280);
    ctx.lineWidth = 22; ctx.lineCap = "round"; ctx.lineJoin = "round"; ctx.strokeStyle = "#fff";
    hint.style.display = "flex";
    }
    resetPad();

    let drawing = false, last = null;
    function pos(e) {
    const r = pad.getBoundingClientRect();
    return { x: (e.clientX - r.left) * 280 / r.width, y: (e.clientY - r.top) * 280 / r.height };
    }
    pad.addEventListener("pointerdown", e => {
    e.preventDefault();
    drawing = true; last = pos(e); hint.style.display = "none";
    try { pad.setPointerCapture(e.pointerId); } catch (_) {}
    ctx.beginPath(); ctx.moveTo(last.x, last.y); ctx.lineTo(last.x + 0.01, last.y); ctx.stroke();
    });
    pad.addEventListener("pointermove", e => {
    if (!drawing) return;
    e.preventDefault();
    const p = pos(e);
    ctx.beginPath(); ctx.moveTo(last.x, last.y); ctx.lineTo(p.x, p.y); ctx.stroke();
    last = p;
    });
    function stop() { if (drawing) { drawing = false; if (auto.checked) run(); } }
    pad.addEventListener("pointerup", stop);
    pad.addEventListener("pointercancel", stop);
    pad.addEventListener("contextmenu", e => e.preventDefault());

  // ----- preprocess like MNIST: crop, fit into 20x20, center of mass at (14,14) -----
    function preprocess() {
    const d = ctx.getImageData(0, 0, 280, 280).data;
    let x0 = 280, y0 = 280, x1 = -1, y1 = -1;
    for (let y = 0; y < 280; y++)
        for (let x = 0; x < 280; x++)
        if (d[(y * 280 + x) * 4] > 50) {
            if (x < x0) x0 = x; if (x > x1) x1 = x;
            if (y < y0) y0 = y; if (y > y1) y1 = y;
        }
    if (x1 < 0) return { empty: true };
    const w = x1 - x0 + 1, h = y1 - y0 + 1;
    if (Math.max(w, h) < MIN_SIZE) return { tooSmall: true };

    const s = 20 / Math.max(w, h);
    const nw = Math.max(1, Math.round(w * s)), nh = Math.max(1, Math.round(h * s));
    const tmp = document.createElement("canvas"); tmp.width = 28; tmp.height = 28;
    const t = tmp.getContext("2d");
    t.fillStyle = "#000"; t.fillRect(0, 0, 28, 28);
    t.imageSmoothingEnabled = true; t.imageSmoothingQuality = "high";
    t.drawImage(pad, x0, y0, w, h, Math.floor((28 - nw) / 2), Math.floor((28 - nh) / 2), nw, nh);

    const img = t.getImageData(0, 0, 28, 28).data;
    let m = 0, cx = 0, cy = 0;
    for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
      const v = img[(y * 28 + x) * 4]; m += v; cx += x * v; cy += y * v;
    }
    const dx = Math.round(14 - cx / m), dy = Math.round(14 - cy / m);
    const out = document.createElement("canvas"); out.width = 28; out.height = 28;
    const o = out.getContext("2d");
    o.fillStyle = "#000"; o.fillRect(0, 0, 28, 28);
    o.drawImage(tmp, dx, dy);

    sctx.drawImage(out, 0, 0);
    const px = o.getImageData(0, 0, 28, 28).data;
    const arr = new Float32Array(784);
    for (let i = 0; i < 784; i++) arr[i] = px[i * 4] / 255;
    return { arr };
    }

    function run() {
    if (typeof MODEL === "undefined") return;
    const r = preprocess();
    if (r.empty) { say("Draw a digit first.", true); return; }
    if (r.tooSmall) {
        resetResult();
        say("Too small to be a digit. Draw it bigger, filling most of the box.", true);
        return;
    }
    const p = predictCNN(r.arr);
    const best = p.indexOf(Math.max(...p));
    const pct = p[best] * 100;
    digitEl.textContent = best;
    digitEl.classList.remove("pop"); void digitEl.offsetWidth; digitEl.classList.add("pop");
    digitEl.classList.toggle("unsure", pct < SURE);
    confEl.textContent = pct.toFixed(1);
    for (let i = 0; i < 10; i++) {
        const b = $("bar" + i);
        b.classList.toggle("top", i === best);
      b.querySelector(".fill").style.width = (p[i] * 100).toFixed(1) + "%";
      b.querySelector(".pc").textContent = (p[i] * 100).toFixed(1) + "%";
    }
    if (pct < SURE) say("Not sure. This guess has low confidence, try a clearer, bigger digit.", true);
    else say("Confident prediction.", false);
    }

    $("predict").addEventListener("click", run);
    $("clear").addEventListener("click", () => {
        resetPad(); sctx.fillStyle = "#000"; sctx.fillRect(0, 0, 28, 28);
        resetResult(); say("Draw a digit to start.", false);
    });

    if (typeof MODEL === "undefined") {
        status.textContent = "weights.js not found. Put weights.js in the same folder as index.html.";
        status.classList.add("bad");
    } else {
        status.textContent = "Model loaded. Runs in your browser.";
        status.classList.add("ok");
    }
})();

if (typeof module !== "undefined") module.exports = { predictCNN };


