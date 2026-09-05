// Lifecycle tests for the browser stream, evaluated from the TypeScript source
// with small Web Audio/WebSocket fakes. The important race is an AudioWorklet
// module finishing after React has already destroyed its owner.

import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { createRequire } from "node:module"
import { fileURLToPath } from "node:url"
import path from "node:path"
import vm from "node:vm"

const require = createRequire(import.meta.url)
const ts = require("typescript")
const here = path.dirname(fileURLToPath(import.meta.url))
const source = readFileSync(path.join(here, "..", "lib", "mrt-stream.ts"), "utf8")
const compiled = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.CommonJS,
    target: ts.ScriptTarget.ES2020,
  },
}).outputText

function deferred() {
  let resolve
  let reject
  const promise = new Promise((yes, no) => {
    resolve = yes
    reject = no
  })
  return { promise, resolve, reject }
}

function loadStream(addModule, initialStorage = {}) {
  const contexts = []
  const sockets = []
  const worklets = []
  const gains = []
  const analysers = []
  const compressors = []
  const timers = new Map()
  const intervals = new Map()
  const storage = new Map(Object.entries(initialStorage))
  let timerId = 0

  function audioParam(value = 0) {
    return {
      value,
      targets: [],
      setTargetAtTime(next, at, constant) {
        this.value = next
        this.targets.push({ next, at, constant })
      },
      cancelScheduledValues() {},
      setValueAtTime(next) {
        this.value = next
      },
      linearRampToValueAtTime(next) {
        this.value = next
      },
    }
  }

  class Node {
    constructor() {
      this.connections = []
      this.disconnected = false
    }
    connect(target) {
      this.connections.push(target)
      return target
    }
    disconnect() {
      this.disconnected = true
      this.connections = []
    }
  }

  class Gain extends Node {
    constructor() {
      super()
      this.gain = audioParam(1)
      gains.push(this)
    }
  }

  class Analyser extends Node {
    constructor() {
      super()
      this.fftSize = 2048
      this.frequencyBinCount = 1024
      this.smoothingTimeConstant = 0
      analysers.push(this)
    }
    getByteTimeDomainData(out) {
      out.fill(128)
    }
    getByteFrequencyData(out) {
      out.fill(0)
    }
    getFloatTimeDomainData(out) {
      // A stable -26 dBFS-ish sine-sized signal for normalizer tests.
      out.fill(0.05)
    }
  }

  class Compressor extends Node {
    constructor() {
      super()
      this.threshold = audioParam(-24)
      this.knee = audioParam(30)
      this.ratio = audioParam(12)
      this.attack = audioParam(0.003)
      this.release = audioParam(0.25)
      this.reduction = 0
      compressors.push(this)
    }
  }

  class FakeAudioContext {
    constructor() {
      this.state = "suspended"
      this.currentTime = 0
      this.destination = new Node()
      this.audioWorklet = { addModule }
      this.closeCalls = 0
      contexts.push(this)
    }
    createGain() {
      return new Gain()
    }
    createAnalyser() {
      return new Analyser()
    }
    createDynamicsCompressor() {
      return new Compressor()
    }
    resume() {
      this.state = "running"
      return Promise.resolve()
    }
    suspend() {
      this.state = "suspended"
      return Promise.resolve()
    }
    close() {
      this.closeCalls += 1
      this.state = "closed"
      return Promise.resolve()
    }
  }

  class FakeWorkletNode extends Node {
    constructor() {
      super()
      this.messages = []
      this.port = {
        onmessage: null,
        closed: false,
        postMessage: (message) => this.messages.push(message),
        close: () => {
          this.port.closed = true
        },
      }
      worklets.push(this)
    }
  }

  class FakeWebSocket {
    static CONNECTING = 0
    static OPEN = 1
    static CLOSING = 2
    static CLOSED = 3
    constructor(url) {
      this.url = url
      this.readyState = FakeWebSocket.CONNECTING
      this.sent = []
      this.closeCalls = 0
      this.onopen = null
      this.onmessage = null
      this.onerror = null
      this.onclose = null
      sockets.push(this)
    }
    send(message) {
      this.sent.push(message)
    }
    close() {
      this.closeCalls += 1
      this.readyState = FakeWebSocket.CLOSED
    }
  }

  const module = { exports: {} }
  const sandbox = {
    module,
    exports: module.exports,
    require,
    process: { env: {} },
    window: {
      location: { protocol: "http:" },
      sessionStorage: {
        getItem: (key) => storage.get(key) ?? null,
        setItem: (key, value) => storage.set(key, value),
        removeItem: (key) => storage.delete(key),
      },
    },
    AudioContext: FakeAudioContext,
    AudioWorkletNode: FakeWorkletNode,
    WebSocket: FakeWebSocket,
    performance: { now: () => 1000 },
    setTimeout: (fn, ms) => {
      const id = ++timerId
      timers.set(id, { fn, ms })
      return id
    },
    clearTimeout: (id) => timers.delete(id),
    setInterval: (fn, ms) => {
      const id = ++timerId
      intervals.set(id, { fn, ms })
      return id
    },
    clearInterval: (id) => intervals.delete(id),
    console,
    ArrayBuffer,
    Int16Array,
    Uint8Array,
    Math,
    Promise,
    Set,
    Error,
  }
  vm.runInNewContext(compiled, sandbox, { filename: "mrt-stream.js" })
  return {
    MrtStream: module.exports.MrtStream,
    contexts,
    sockets,
    worklets,
    gains,
    analysers,
    compressors,
    timers,
    intervals,
    storage,
    WebSocket: FakeWebSocket,
  }
}

const results = []
const CONTROLS = Object.freeze({ station: "dusty-beats", drums: true })
async function test(name, exercise) {
  try {
    await exercise()
    results.push(true)
    console.log(`PASS  ${name}`)
  } catch (error) {
    results.push(false)
    console.error(`FAIL  ${name}`)
    console.error(error)
  }
}

await test("destroy prevents a deferred sink from resurrecting", async () => {
  const moduleLoad = deferred()
  const runtime = loadStream(() => moduleLoad.promise)
  const states = []
  const stream = new runtime.MrtStream((state) => states.push(state))

  const starting = stream.start(CONTROLS)
  assert.equal(runtime.contexts.length, 1)
  assert.equal(runtime.sockets.length, 1)

  const destroyed = stream.destroy()
  moduleLoad.resolve()
  await Promise.all([starting, destroyed])

  assert.equal(runtime.worklets.length, 0)
  assert.equal(runtime.intervals.size, 0)
  assert.equal(runtime.timers.size, 0)
  assert.equal(runtime.contexts[0].state, "closed")
  assert.equal(runtime.contexts[0].closeCalls, 1)
  assert.equal(runtime.sockets[0].closeCalls, 1)
  assert.equal(runtime.sockets[0].onclose, null)
  assert.equal(states.length, 1, "destroyed continuation emitted state")

  await stream.destroy()
  assert.equal(runtime.contexts[0].closeCalls, 1, "destroy was not idempotent")
})

await test("hello and live updates use the minimal listener protocol", async () => {
  const runtime = loadStream(() => Promise.resolve())
  const stream = new runtime.MrtStream(() => {})
  const controls = {
    station: "rainy-piano",
    drums: false,
  }
  const starting = stream.start(controls)
  const socket = runtime.sockets[0]
  socket.readyState = runtime.WebSocket.OPEN
  socket.onopen()
  await starting

  assert.deepEqual(JSON.parse(socket.sent[0]), {
    type: "hello",
    sessionId: null,
    ...controls,
  })

  const changed = {
    station: "jazz-cafe",
    drums: true,
  }
  stream.setControls(changed)
  const messages = socket.sent.slice(1).map((message) => JSON.parse(message))
  assert.deepEqual(messages, [{ type: "controls", ...changed }])

  await stream.destroy()
})

await test("a new take drops old PCM until its fresh-session acknowledgement", async () => {
  const runtime = loadStream(() => Promise.resolve())
  const states = []
  const stream = new runtime.MrtStream((state) => states.push(state))
  const starting = stream.start(CONTROLS)
  const socket = runtime.sockets[0]
  socket.readyState = runtime.WebSocket.OPEN
  socket.onopen()
  await starting

  socket.onmessage({
    data: JSON.stringify({
      type: "hello",
      sessionId: "old-session",
      resumed: false,
      sampleRate: 48000,
      channels: 2,
    }),
  })
  const worklet = runtime.worklets[0]
  worklet.messages.length = 0

  stream.newVariation()
  assert.equal(JSON.parse(socket.sent.at(-1)).type, "variation")
  assert.equal(runtime.storage.has("lofai.sessionId"), false)
  assert.equal(states.at(-1).variationPending, true)
  assert.equal(worklet.messages.at(-1).type, "reset")

  const pcmBefore = worklet.messages.filter((message) => message.type === "pcm").length
  socket.onmessage({ data: new ArrayBuffer(32) })
  assert.equal(
    worklet.messages.filter((message) => message.type === "pcm").length,
    pcmBefore,
    "pre-ack PCM leaked into the new take",
  )

  socket.onmessage({
    data: JSON.stringify({ type: "variation", sessionId: "fresh-session", seed: 42 }),
  })
  assert.equal(states.at(-1).variationPending, false)
  assert.equal(JSON.parse(runtime.storage.get("lofai.sessionId")).id, "fresh-session")

  socket.onmessage({ data: new ArrayBuffer(32) })
  assert.equal(
    worklet.messages.filter((message) => message.type === "pcm").length,
    pcmBefore + 1,
    "post-ack PCM was not accepted",
  )

  await stream.destroy()
})

await test("stale session IDs are not replayed and a stale server echo requests a variation", async () => {
  const expired = JSON.stringify({
    version: 1,
    id: "expired-session",
    savedAt: Date.now() - 10 * 60 * 1000,
  })
  const expiredRuntime = loadStream(() => Promise.resolve(), { "lofai.sessionId": expired })
  const expiredStream = new expiredRuntime.MrtStream(() => {})
  const expiredStart = expiredStream.start(CONTROLS)
  const expiredSocket = expiredRuntime.sockets[0]
  expiredSocket.readyState = expiredRuntime.WebSocket.OPEN
  expiredSocket.onopen()
  await expiredStart
  assert.equal(JSON.parse(expiredSocket.sent[0]).sessionId, null)
  await expiredStream.destroy()

  const current = JSON.stringify({ version: 1, id: "stale-echo", savedAt: Date.now() })
  const runtime = loadStream(() => Promise.resolve(), { "lofai.sessionId": current })
  const stream = new runtime.MrtStream(() => {})
  const starting = stream.start(CONTROLS)
  const socket = runtime.sockets[0]
  socket.readyState = runtime.WebSocket.OPEN
  socket.onopen()
  await starting
  assert.equal(JSON.parse(socket.sent[0]).sessionId, "stale-echo")

  socket.onmessage({
    data: JSON.stringify({
      type: "hello",
      sessionId: "stale-echo",
      resumed: false,
      sampleRate: 48000,
      channels: 2,
    }),
  })
  assert.equal(JSON.parse(socket.sent.at(-1)).type, "variation")
  assert.equal(runtime.storage.has("lofai.sessionId"), false)

  await stream.destroy()
})

await test("mastering raises quiet material slowly and keeps volume after the limiter", async () => {
  const runtime = loadStream(() => Promise.resolve())
  const stream = new runtime.MrtStream(() => {})
  const starting = stream.start(CONTROLS)
  const socket = runtime.sockets[0]
  socket.readyState = runtime.WebSocket.OPEN
  socket.onopen()
  await starting

  assert.equal(runtime.compressors.length, 1)
  const limiter = runtime.compressors[0]
  assert.equal(limiter.threshold.value, -2.5)
  assert.equal(limiter.knee.value, 0)
  assert.equal(limiter.ratio.value, 20)
  assert.equal(limiter.attack.value, 0.003)
  assert.equal(limiter.release.value, 0.3)

  // input meter -> slow gain -> limiter -> -1 dB ceiling -> user volume -> output meter
  assert.equal(runtime.analysers[0].connections[0], runtime.gains[0])
  assert.equal(runtime.gains[0].connections[0], limiter)
  assert.equal(limiter.connections[0], runtime.gains[1])
  assert.equal(runtime.gains[1].connections[0], runtime.gains[2])
  assert.equal(runtime.gains[2].connections[0], runtime.analysers[1])
  assert.ok(Math.abs(runtime.gains[1].gain.value - 10 ** (-1 / 20)) < 1e-9)

  socket.onmessage({
    data: JSON.stringify({ type: "status", state: "active", realtimeFactor: 1.2 }),
  })
  const loudnessTick = [...runtime.intervals.values()][0].fn
  loudnessTick()
  loudnessTick()
  loudnessTick()
  assert.ok(runtime.gains[0].gain.value > 1, "quiet material did not receive slow makeup")
  assert.ok(runtime.gains[0].gain.value < 1.2, "normalizer jumped instead of moving slowly")

  stream.setVolume(0.4)
  assert.equal(runtime.gains[2].gain.value, 0.4)
  await stream.destroy()
  assert.equal(runtime.intervals.size, 0)
})

await test("server restart preserves the stream object and schedules reconnect", async () => {
  const runtime = loadStream(() => Promise.resolve())
  const states = []
  const stream = new runtime.MrtStream((state) => states.push(state))
  const starting = stream.start(CONTROLS)
  const socket = runtime.sockets[0]
  socket.readyState = runtime.WebSocket.OPEN
  socket.onopen()
  await starting
  assert.equal(runtime.worklets.length, 1)

  socket.readyState = runtime.WebSocket.CLOSED
  socket.onclose({ code: 1012, reason: "service restart" })

  assert.equal(runtime.timers.size, 1)
  assert.equal(runtime.intervals.size, 1)
  assert.equal(runtime.contexts[0].state, "running")
  assert.equal(runtime.worklets[0].port.closed, false)
  assert.equal(runtime.worklets[0].messages.at(-1).type, "reset")
  assert.equal(states.at(-1).message, "reconnecting")

  await stream.destroy()
  assert.equal(runtime.timers.size, 0)
  assert.equal(runtime.intervals.size, 0)
  assert.equal(runtime.contexts[0].state, "closed")
})

await test("generation errors stop intent and survive the following suspended status", async () => {
  const runtime = loadStream(() => Promise.resolve())
  const states = []
  const stream = new runtime.MrtStream((state) => states.push(state))
  const controls = {
    station: "dusty-beats",
    drums: true,
  }
  const starting = stream.start(controls)
  const socket = runtime.sockets[0]
  socket.readyState = runtime.WebSocket.OPEN
  socket.onopen()
  await starting

  socket.onmessage({
    data: JSON.stringify({ type: "error", message: "native generation failed" }),
  })
  assert.equal(states.at(-1).status, "error")
  assert.equal(runtime.worklets[0].messages.at(-1).type, "reset")

  socket.onmessage({
    data: JSON.stringify({ type: "status", state: "suspended", realtimeFactor: 1.2 }),
  })
  assert.equal(states.at(-1).status, "error")
  assert.equal(states.at(-1).message, "native generation failed")

  await stream.start(controls)
  assert.equal(JSON.parse(socket.sent.at(-1)).type, "resume")
  await stream.destroy()
})

await test("terminal model load failure preserves its message without reconnecting", async () => {
  const runtime = loadStream(() => Promise.resolve())
  const states = []
  const stream = new runtime.MrtStream((state) => states.push(state))
  const starting = stream.start(CONTROLS)
  const socket = runtime.sockets[0]
  socket.readyState = runtime.WebSocket.OPEN
  socket.onopen()
  await starting

  socket.onmessage({
    data: JSON.stringify({
      type: "error",
      message: "Metal compiler unavailable",
      terminal: true,
    }),
  })
  socket.readyState = runtime.WebSocket.CLOSED
  socket.onclose({ code: 1011, reason: "model failed to load" })

  assert.equal(runtime.timers.size, 0)
  assert.equal(states.at(-1).status, "error")
  assert.equal(states.at(-1).message, "Metal compiler unavailable")
  assert.equal(runtime.contexts[0].state, "running", "stream object was destroyed")

  await stream.start(CONTROLS)
  assert.equal(runtime.sockets.length, 2, "the existing stream object could not retry")
  await stream.destroy()
})

await test("server shutdown disposes a paused audio graph too", async () => {
  const runtime = loadStream(() => Promise.resolve())
  const stream = new runtime.MrtStream(() => {})
  const starting = stream.start(CONTROLS)
  const socket = runtime.sockets[0]
  socket.readyState = runtime.WebSocket.OPEN
  socket.onopen()
  await starting

  stream.pause()
  assert.equal(runtime.timers.size, 1, "pause did not schedule context suspension")
  socket.readyState = runtime.WebSocket.CLOSED
  socket.onclose({ code: 1012 })
  await stream.destroy()

  assert.equal(runtime.timers.size, 0)
  assert.equal(runtime.intervals.size, 0)
  assert.equal(runtime.contexts[0].state, "closed")
  assert.equal(runtime.worklets[0].port.closed, true)
  assert.equal(runtime.worklets[0].disconnected, true)
})

const failed = results.filter((ok) => !ok).length
console.log(`\n${results.length - failed}/${results.length} passed`)
process.exit(failed ? 1 : 0)
