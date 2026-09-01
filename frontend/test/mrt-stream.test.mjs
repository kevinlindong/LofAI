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

function loadStream(addModule) {
  const contexts = []
  const sockets = []
  const worklets = []
  const timers = new Map()
  const intervals = new Map()
  let timerId = 0

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
      this.gain = {
        value: 1,
        setTargetAtTime() {},
        cancelScheduledValues() {},
        setValueAtTime() {},
        linearRampToValueAtTime() {},
      }
    }
  }

  class Analyser extends Node {
    constructor() {
      super()
      this.fftSize = 2048
      this.frequencyBinCount = 1024
      this.smoothingTimeConstant = 0
    }
    getByteTimeDomainData(out) {
      out.fill(128)
    }
    getByteFrequencyData(out) {
      out.fill(0)
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
      sessionStorage: { getItem: () => null, setItem() {} },
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
    timers,
    intervals,
    WebSocket: FakeWebSocket,
  }
}

const results = []
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

  const starting = stream.start("neutral", "guitar")
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

await test("intentional server shutdown never schedules reconnect processing", async () => {
  const runtime = loadStream(() => Promise.resolve())
  const states = []
  const stream = new runtime.MrtStream((state) => states.push(state))
  const starting = stream.start("neutral", "guitar")
  const socket = runtime.sockets[0]
  socket.readyState = runtime.WebSocket.OPEN
  socket.onopen()
  await starting
  assert.equal(runtime.worklets.length, 1)

  socket.readyState = runtime.WebSocket.CLOSED
  socket.onclose({ code: 1012 })
  await stream.destroy()

  assert.equal(runtime.timers.size, 0)
  assert.equal(runtime.intervals.size, 0)
  assert.equal(runtime.contexts[0].state, "closed")
  assert.equal(runtime.worklets[0].port.closed, true)
  assert.equal(runtime.worklets[0].disconnected, true)
  assert.equal(states.at(-1).message, "application stopped")
})

await test("server shutdown disposes a paused audio graph too", async () => {
  const runtime = loadStream(() => Promise.resolve())
  const stream = new runtime.MrtStream(() => {})
  const starting = stream.start("neutral", "guitar")
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
