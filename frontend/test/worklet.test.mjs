// exercise the worklet's dsp outside a browser: stub the three globals an
// AudioWorkletProcessor gets, then drive it with known audio and look at what
// comes out. this is the code that has to never click, so it gets tested.
import { readFileSync } from "node:fs"
import { fileURLToPath } from "node:url"
import path from "node:path"
import vm from "node:vm"

const SR = 48000
const Q = 128
const here = path.dirname(fileURLToPath(import.meta.url))
const source = readFileSync(path.join(here, "..", "public", "mrt-pcm-worklet.js"), "utf8")

let Registered = null
const sandbox = {
  sampleRate: SR,
  registerProcessor: (_name, cls) => {
    Registered = cls
  },
  AudioWorkletProcessor: class {
    constructor() {
      const outbox = []
      this.port = {
        onmessage: null,
        postMessage: (msg) => outbox.push(msg),
        close: () => {
          this.port._closed = true
        },
        _outbox: outbox,
        _closed: false,
      }
    }
  },
  console,
}
vm.createContext(sandbox)
vm.runInContext(source, sandbox)

function make(opts = {}) {
  const p = new Registered({
    processorOptions: {
      channels: 2,
      sourceRate: SR,
      ringSeconds: 6,
      prebufferSeconds: 0.5,
      rebufferSeconds: 0.4,
      comfortSeconds: 1.0,
      minRate: 0.95,
      fadeSeconds: 0.02,
      ...opts,
    },
  })
  return p
}

// a stereo sine, phase-continuous across calls
function sine(frames, phase, freq = 440, amp = 0.5) {
  const out = new Float32Array(frames * 2)
  for (let i = 0; i < frames; i++) {
    const v = Math.sin(phase + (2 * Math.PI * freq * i) / SR) * amp
    out[i * 2] = v
    out[i * 2 + 1] = v
  }
  return { data: out, phase: phase + (2 * Math.PI * freq * frames) / SR }
}

function feed(p, frames, state, opts) {
  const s = sine(frames, state.phase, opts?.freq, opts?.amp)
  state.phase = s.phase
  const pcm = new Int16Array(s.data.length)
  for (let i = 0; i < s.data.length; i++) pcm[i] = Math.round(s.data[i] * 32767)
  p.port.onmessage({ data: { type: "pcm", pcm: pcm.buffer } })
}

function pull(p, quanta) {
  const left = []
  const right = []
  for (let i = 0; i < quanta; i++) {
    const out = [new Float32Array(Q), new Float32Array(Q)]
    p.process([], [out])
    left.push(...out[0])
    right.push(...out[1])
  }
  return [Float32Array.from(left), Float32Array.from(right)]
}

// a splice shows up as a spike in the second difference; a pure sine's is tiny
function worstSecondDiff(x, from = 0, to = x.length) {
  let worst = 0
  let at = -1
  for (let i = from + 2; i < to; i++) {
    const d = Math.abs(x[i] - 2 * x[i - 1] + x[i - 2])
    if (d > worst) {
      worst = d
      at = i
    }
  }
  return { worst, at }
}

const results = []
const check = (name, ok, detail) => {
  results.push({ name, ok, detail })
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? "  — " + detail : ""}`)
}

// theoretical bound for a clean 440Hz sine at amplitude 0.5
const w = (2 * Math.PI * 440) / SR
const SINE_BOUND = 0.5 * w * w // ~0.00169

// --- 1. steady playback is one continuous signal -------------------------
{
  const p = make()
  const st = { phase: 0 }
  feed(p, SR * 3, st) // 3s banked
  p.port.onmessage({ data: { type: "play", fresh: true } })
  const [l] = pull(p, Math.floor((SR * 1.5) / Q))

  // skip the 20ms fade-in
  const after = Math.ceil(0.03 * SR)
  const { worst, at } = worstSecondDiff(l, after)
  check(
    "steady playback has no discontinuity",
    worst < SINE_BOUND * 3,
    `worst 2nd-diff ${worst.toExponential(2)} at ${at} (sine bound ${SINE_BOUND.toExponential(2)})`,
  )

  let peak = 0
  for (let i = after; i < l.length; i++) peak = Math.max(peak, Math.abs(l[i]))
  check("amplitude preserved", Math.abs(peak - 0.5) < 0.02, `peak ${peak.toFixed(4)}`)
}

// --- 2. it waits for the prebuffer, then fades in ------------------------
{
  const p = make()
  const st = { phase: 0 }
  feed(p, Math.floor(SR * 0.2), st) // less than the 0.5s prebuffer
  p.port.onmessage({ data: { type: "play", fresh: true } })
  const [a] = pull(p, 20)
  const silent = a.every((v) => v === 0)

  feed(p, Math.floor(SR * 1.0), st)
  const [b] = pull(p, Math.floor((0.02 * SR) / Q))
  let monotonic = true
  let envelope = 0
  for (let i = 0; i < b.length; i++) {
    const m = Math.abs(b[i])
    if (m > envelope) envelope = m
  }
  // the fade should have started from zero, not jumped
  const startsQuiet = Math.abs(b[0]) < 0.02
  check("holds silence below the prebuffer", silent, "")
  check("fades in from zero", startsQuiet && envelope > 0.05, `first ${b[0].toFixed(4)}, reached ${envelope.toFixed(3)}`)
}

// --- 3. running dry fades out instead of cutting -------------------------
{
  const p = make()
  const st = { phase: 0 }
  feed(p, Math.floor(SR * 0.8), st)
  p.port.onmessage({ data: { type: "play", fresh: true } })
  p.port._outbox.length = 0
  const [l] = pull(p, Math.floor((SR * 1.2) / Q))

  const { worst, at } = worstSecondDiff(l, Math.ceil(0.03 * SR))
  const starved = p.port._outbox.some((m) => m.type === "starved")
  // find where it actually goes quiet and confirm it got there by a ramp
  let lastLoud = 0
  for (let i = 0; i < l.length; i++) if (Math.abs(l[i]) > 0.01) lastLoud = i
  const rampSamples = (() => {
    let above = 0
    for (let i = lastLoud; i > 0 && Math.abs(l[i]) < 0.45; i--) above = lastLoud - i
    return above
  })()

  check("underrun reported", starved, "")
  check(
    "underrun fades rather than cuts",
    worst < SINE_BOUND * 3,
    `worst 2nd-diff ${worst.toExponential(2)} at ${at}, ramp ~${rampSamples} samples`,
  )
}

// --- 4. refill resumes without a splice ----------------------------------
{
  const p = make()
  const st = { phase: 0 }
  feed(p, Math.floor(SR * 0.8), st)
  p.port.onmessage({ data: { type: "play", fresh: true } })
  pull(p, Math.floor((SR * 1.0) / Q)) // drain it
  feed(p, Math.floor(SR * 1.5), st) // refill
  const [l] = pull(p, Math.floor((SR * 0.8) / Q))
  const { worst, at } = worstSecondDiff(l)
  check(
    "refill resumes without a splice",
    worst < SINE_BOUND * 3,
    `worst 2nd-diff ${worst.toExponential(2)} at ${at}`,
  )
}

// --- 5. rate stretching stays continuous ---------------------------------
{
  const p = make({ comfortSeconds: 4, minRate: 0.9, prebufferSeconds: 0.3 })
  const st = { phase: 0 }
  feed(p, Math.floor(SR * 0.5), st)
  p.port.onmessage({ data: { type: "play", fresh: true } })
  // keep it just barely fed so the rate controller stays engaged
  const chunks = []
  for (let i = 0; i < 200; i++) {
    if (i % 4 === 0) feed(p, Math.floor(SR * 0.02), st)
    const out = [new Float32Array(Q), new Float32Array(Q)]
    p.process([], [out])
    chunks.push(...out[0])
  }
  const l = Float32Array.from(chunks)
  const { worst, at } = worstSecondDiff(l, Math.ceil(0.05 * SR))
  check(
    "rate stretching stays continuous",
    worst < SINE_BOUND * 4,
    `rate ${p.rate.toFixed(4)}, worst 2nd-diff ${worst.toExponential(2)} at ${at}`,
  )
  check("rate stayed within its clamp", p.rate >= 0.9 - 1e-6 && p.rate <= 1 + 1e-6, `rate ${p.rate.toFixed(4)}`)
}

// --- 6. a context that refused 48k gets resampled ------------------------
{
  const p = make({ sourceRate: 44100 })
  const st = { phase: 0 }
  // 44.1k source: generate at that rate
  const frames = 44100 * 2
  const data = new Int16Array(frames * 2)
  for (let i = 0; i < frames; i++) {
    const v = Math.sin((2 * Math.PI * 440 * i) / 44100) * 0.5
    data[i * 2] = Math.round(v * 32767)
    data[i * 2 + 1] = Math.round(v * 32767)
  }
  p.port.onmessage({ data: { type: "pcm", pcm: data.buffer } })
  p.port.onmessage({ data: { type: "play", fresh: true } })
  const [l] = pull(p, Math.floor((SR * 1.0) / Q))
  const after = Math.ceil(0.05 * SR)
  // count zero crossings to recover the pitch: it must still be 440Hz
  let crossings = 0
  for (let i = after + 1; i < l.length; i++) {
    if (l[i - 1] < 0 && l[i] >= 0) crossings++
  }
  const seconds = (l.length - after) / SR
  const freq = crossings / seconds
  check(
    "resamples a non-48k context to the right pitch",
    Math.abs(freq - 440) < 5,
    `measured ${freq.toFixed(1)}Hz`,
  )
}

// --- 7. the ring wraps ---------------------------------------------------
{
  const p = make({ ringSeconds: 1.5, prebufferSeconds: 0.2 })
  const st = { phase: 0 }
  feed(p, Math.floor(SR * 0.5), st)
  p.port.onmessage({ data: { type: "play", fresh: true } })
  const all = []
  for (let i = 0; i < 400; i++) {
    if (i % 20 === 0) feed(p, Math.floor(SR * 0.07), st)
    const out = [new Float32Array(Q), new Float32Array(Q)]
    p.process([], [out])
    all.push(...out[0])
  }
  const l = Float32Array.from(all)
  const { worst, at } = worstSecondDiff(l, Math.ceil(0.05 * SR))
  check(
    "ring wraparound is seamless",
    worst < SINE_BOUND * 4,
    `wrote ${(p.written / SR).toFixed(2)}s through a ${(p.capacity / SR).toFixed(2)}s ring, worst 2nd-diff ${worst.toExponential(2)} at ${at}`,
  )
}

// --- 8. resetting a live stream fades before clearing -------------------
{
  const p = make({ prebufferSeconds: 0.2 })
  const st = { phase: 0 }
  feed(p, Math.floor(SR * 1.0), st)
  p.port.onmessage({ data: { type: "play", fresh: true } })
  pull(p, Math.floor((SR * 0.25) / Q))
  p.port.onmessage({ data: { type: "reset" } })
  const [l] = pull(p, Math.floor((SR * 0.1) / Q))
  const { worst, at } = worstSecondDiff(l)
  const tail = l.subarray(Math.max(0, l.length - Q))
  check(
    "live reset fades without a click",
    worst < SINE_BOUND * 4 && tail.every((v) => v === 0),
    `worst 2nd-diff ${worst.toExponential(2)} at ${at}`,
  )
}

// --- 9. disposal ends audio-thread processing ---------------------------
{
  const p = make()
  const st = { phase: 0 }
  feed(p, Math.floor(SR * 0.5), st)
  p.port._outbox.length = 0
  p.port.onmessage({ data: { type: "dispose" } })

  const out = [new Float32Array(Q), new Float32Array(Q)]
  const keepAlive = p.process([], [out])
  const reports = p.port._outbox.length
  // A repeated command can race with the final render quantum; it must remain
  // a harmless no-op even though the port handler has already been detached.
  p.receive({ type: "dispose" })

  check("dispose closes the worklet port", p.port._closed, "")
  check("dispose releases the PCM ring", p.ring.length === 0, "")
  check("dispose terminates audio-thread processing", keepAlive === false, "")
  check("disposed worklet emits no more reports", reports === 0, "")
}

// PCM and the next play command may reach the audio thread before its reset
// fade finishes. The new take's first packet must survive that boundary.
{
  const p = make({ prebufferSeconds: 0.2, minRate: 1 })
  const old = new Int16Array(SR * 2).fill(12000)
  p.receive({ type: "pcm", pcm: old.buffer })
  p.receive({ type: "play", fresh: true })
  pull(p, 30)
  p.receive({ type: "reset" })
  p.receive({ type: "play", fresh: true })
  const fresh = new Int16Array(SR).fill(-12000)
  p.receive({ type: "pcm", pcm: fresh.buffer })
  const [out] = pull(p, 40)
  check("reset preserves PCM received during its fade", p.written === SR / 2 && out.at(-1) < -0.3, "")
  check("new take follows the reset fade without a splice", worstSecondDiff(out).worst < 0.002, "")
}

// Pause must win even when a reset is still fading and new PCM is queued.
{
  const p = make({ prebufferSeconds: 0.2, minRate: 1 })
  feed(p, SR, { phase: 0 })
  check("buffered PCM stays silent until play", pull(p, 12)[0].every((v) => v === 0), "")
  p.receive({ type: "play", fresh: true })
  pull(p, 30)
  p.receive({ type: "reset" })
  p.receive({ type: "play", fresh: true })
  p.receive({ type: "stop" })
  feed(p, SR / 2, { phase: 0 })
  pull(p, 15)
  check("pause during reset keeps the new reservoir silent", pull(p, 30)[0].every((v) => v === 0), "")
  p.receive({ type: "play", fresh: false })
  check("paused reset retains audio for resume", pull(p, 30)[0].some((v) => Math.abs(v) > 0.1), "")
}

// A frozen tab can overflow. Dropping one chunk and accepting the next used
// to create a full-amplitude splice after the old reservoir drained.
{
  const p = make({ ringSeconds: 1, prebufferSeconds: 0.2, minRate: 1 })
  p.receive({ type: "pcm", pcm: new Int16Array(Math.floor(SR * 0.8) * 2).fill(12000).buffer })
  p.receive({ type: "play", fresh: true })
  pull(p, 30)
  p.receive({ type: "pcm", pcm: new Int16Array(Math.floor(SR * 0.6) * 2).fill(-12000).buffer })
  const [out] = pull(p, 40)
  check("overflow recovers a bounded contiguous reservoir", p.overflows === 1 && p.written === SR * 0.6, "")
  check("overflow fades into new audio without a splice", out.at(-1) < -0.3 && worstSecondDiff(out).worst < 0.002, "")
}

// At matching rates the transport must preserve every PCM sample exactly.
{
  const p = make({ prebufferSeconds: 0.2, minRate: 1 })
  const pcm = new Int16Array(SR * 2)
  for (let i = 0; i < pcm.length; i++) pcm[i] = (i * 7919) % 65536 - 32768
  p.receive({ type: "pcm", pcm: pcm.buffer })
  p.receive({ type: "play", fresh: true })
  pull(p, 30)
  const start = p.readPos
  const [left, right] = pull(p, 10)
  check(
    "native-rate rendering preserves stereo PCM exactly",
    left.every((v, i) => v === pcm[(start + i) * 2] / 32768) &&
      right.every((v, i) => v === pcm[(start + i) * 2 + 1] / 32768),
    "",
  )
}

{
  const p = make({ prebufferSeconds: 0.2, rebufferSeconds: 0.6, minRate: 1 })
  feed(p, SR / 2, { phase: 0 })
  p.receive({ type: "play", fresh: true })
  pull(p, Math.ceil(SR / Q))
  p.receive({ type: "config", prebufferSeconds: 0.1, rebufferSeconds: 0.6 })
  check("throughput updates preserve the rebuffer threshold after underrun", p.need === SR * 0.6, "")
}

const failed = results.filter((r) => !r.ok)
console.log(`\n${results.length - failed.length}/${results.length} passed`)
process.exit(failed.length ? 1 : 0)
