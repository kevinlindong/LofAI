// websocket client for the magenta realtime 2 backend
//
// the backend sends a continuous stream of interleaved 16-bit pcm chunks over a
// websocket. an <audio> element cannot play that - it wants a container format
// with a known length - so we feed the chunks into an AudioWorklet that owns a
// ring buffer and one fractional read cursor, and it turns them into a single
// continuous signal on the audio thread.
//
// that last part is the whole point. the obvious approach - one
// AudioBufferSourceNode per chunk, scheduled from a timer - fails twice over:
// the timer runs on the main thread, so a busy tab schedules late and the
// stream clicks, and each source resamples from its own phase, so any playback
// rate other than 1.0 puts a discontinuity at every chunk seam. on this stream
// that was a tick two or three times a second. the worklet has no seams to
// click at and cannot be starved by rendering work.
//
// the reservoir is still here, because generation speed varies by machine, but
// it is sized from the real time factor the backend measures rather than
// assumed to be needed.

export type StreamStatus =
  | "idle"
  | "connecting"
  | "loading"
  | "queued"
  | "buffering"
  | "live"
  | "paused"
  | "error"

export interface StreamState {
  status: StreamStatus
  queuePosition: number
  listeners: number
  capacity: number
  message: string | null
  // 0..1 progress while filling the reservoir, for the UI
  bufferProgress: number
}

interface Reservoir {
  prebufferSeconds: number
  rebufferSeconds: number
  comfortSeconds: number
  minRate: number
}

// how deep to bank before playing, as a function of how much faster than real
// time the backend says it is rendering.
//
// a machine with headroom needs only enough to cover jitter, and making that
// listener wait five seconds for their first note is pure loss. a machine that
// cannot hold real time gets a deeper bank, but playback stays at 1.0x in every
// tier so the melody never changes pitch as that bank moves.
function reservoirFor(realtimeFactor: number): Reservoir {
  if (realtimeFactor <= 0) {
    // nothing measured yet - the backend has not rendered for us
    return { prebufferSeconds: 2.5, rebufferSeconds: 2, comfortSeconds: 4, minRate: 1 }
  }
  if (realtimeFactor >= 1.15) {
    return { prebufferSeconds: 1.0, rebufferSeconds: 1.0, comfortSeconds: 2.5, minRate: 1 }
  }
  if (realtimeFactor >= 1.0) {
    return { prebufferSeconds: 3, rebufferSeconds: 2.5, comfortSeconds: 5, minRate: 1 }
  }
  return { prebufferSeconds: 6, rebufferSeconds: 5, comfortSeconds: 8, minRate: 1 }
}

const WORKLET_URL = "/mrt-pcm-worklet.js"
const RING_SECONDS = 45
const TERMINAL_SERVER_CLOSE_CODES = new Set([1000, 1001, 1012])

class SinkBuildCancelled extends Error {}

// sessionStorage rather than localStorage: it survives a reload but is scoped
// to the tab, so a second tab opens a second stream instead of both trying to
// resume the same one
const SESSION_KEY = "lofai.sessionId"

function backendHost(): string {
  return process.env.NEXT_PUBLIC_BACKEND_HOST ?? "localhost:8000"
}

function readSessionId(): string | null {
  try {
    return window.sessionStorage.getItem(SESSION_KEY)
  } catch {
    return null
  }
}

function writeSessionId(id: string) {
  try {
    window.sessionStorage.setItem(SESSION_KEY, id)
  } catch {
    // private browsing - the session just won't survive a reload
  }
}

// what the stream needs from whatever is actually making sound
interface PcmSink {
  readonly output: AudioNode
  push(pcm: ArrayBuffer): void
  play(fresh: boolean): void
  stop(): void
  reset(): void
  configure(reservoir: Reservoir): void
  dispose(): void
}

interface SinkReport {
  playing: boolean
  buffered: number
  need: number
}

// --- the real one: a ring buffer on the audio thread ---

class WorkletSink implements PcmSink {
  private disposed = false

  constructor(
    private node: AudioWorkletNode,
    onReport: (report: SinkReport) => void,
    onStarved: () => void,
  ) {
    node.port.onmessage = (event) => {
      const data = event.data
      if (data.type === "starved") onStarved()
      else if (data.type === "state") onReport(data as SinkReport)
    }
  }

  get output(): AudioNode {
    return this.node
  }

  push(pcm: ArrayBuffer) {
    // Keep the network's int16 representation all the way into the worklet.
    // The transfer is zero-copy and conversion happens only for samples that
    // actually reach the speakers.
    this.node.port.postMessage({ type: "pcm", pcm }, [pcm])
  }

  play(fresh: boolean) {
    this.node.port.postMessage({ type: "play", fresh })
  }

  stop() {
    this.node.port.postMessage({ type: "stop" })
  }

  reset() {
    this.node.port.postMessage({ type: "reset" })
  }

  configure(reservoir: Reservoir) {
    this.node.port.postMessage({ type: "config", ...reservoir })
  }

  dispose() {
    if (this.disposed) return
    this.disposed = true
    this.node.port.onmessage = null
    this.node.port.postMessage({ type: "dispose" })
    this.node.port.close()
    this.node.disconnect()
  }
}

// --- the fallback: chunk scheduling from a timer ---
//
// only used where AudioWorklet is missing or its module will not load. it has
// the seam problem described at the top of the file, so it plays at a flat 1.0
// and simply refills when it runs dry rather than stretching - a longer pause,
// but no ticking.

class TimerSink implements PcmSink {
  private gate: GainNode
  private queue: AudioBuffer[] = []
  private queuedSeconds = 0
  private scheduled: AudioBufferSourceNode[] = []
  private nextStart = 0
  private playing = false
  private want = false
  private need: number
  private ticker: ReturnType<typeof setInterval>
  private reservoir = reservoirFor(0)
  private disposed = false

  constructor(
    private ctx: AudioContext,
    private channels: number,
    // the stream's rate, not the context's: on a device that refused 48k these
    // differ, and declaring the buffer at the source rate is what makes the
    // browser resample it instead of playing it sharp
    private sourceRate: number,
    private onReport: (report: SinkReport) => void,
    private onStarved: () => void,
  ) {
    this.gate = ctx.createGain()
    this.gate.gain.value = 0
    this.need = this.reservoir.prebufferSeconds
    this.ticker = setInterval(() => this.tick(), 60)
  }

  get output(): AudioNode {
    return this.gate
  }

  push(pcm: ArrayBuffer) {
    if (this.disposed) return
    const samples = new Int16Array(pcm)
    const frames = Math.floor(samples.length / this.channels)
    if (frames === 0) return
    const buffer = this.ctx.createBuffer(this.channels, frames, this.sourceRate)
    for (let channel = 0; channel < this.channels; channel++) {
      const target = buffer.getChannelData(channel)
      for (let frame = 0; frame < frames; frame++) {
        target[frame] = samples[frame * this.channels + channel] / 32768
      }
    }
    this.queue.push(buffer)
    this.queuedSeconds += buffer.duration
  }

  play(fresh: boolean) {
    if (this.disposed) return
    this.want = true
    if (!this.playing) {
      this.need = fresh
        ? this.reservoir.prebufferSeconds
        : this.reservoir.rebufferSeconds
    }
  }

  stop() {
    if (this.disposed) return
    this.want = false
    this.fadeTo(0)
    this.playing = false
  }

  reset() {
    if (this.disposed) return
    this.stopScheduled()
    this.queue = []
    this.queuedSeconds = 0
    this.playing = false
    this.need = this.reservoir.prebufferSeconds
  }

  configure(reservoir: Reservoir) {
    if (this.disposed) return
    this.reservoir = reservoir
    if (!this.playing) this.need = reservoir.prebufferSeconds
  }

  dispose() {
    if (this.disposed) return
    this.disposed = true
    clearInterval(this.ticker)
    this.want = false
    this.playing = false
    this.queue = []
    this.queuedSeconds = 0
    for (const source of this.scheduled) {
      source.onended = null
      try {
        source.stop()
      } catch {
        // already finished
      }
      source.disconnect()
    }
    this.scheduled = []
    this.nextStart = 0
    this.gate.disconnect()
  }

  private buffered(): number {
    return this.queuedSeconds + Math.max(0, this.nextStart - this.ctx.currentTime)
  }

  private tick() {
    if (this.disposed) return
    const buffered = this.buffered()
    if (this.want) {
      if (!this.playing) {
        if (buffered >= this.need) {
          this.playing = true
          this.nextStart = this.ctx.currentTime + 0.05
          this.fadeTo(1)
        }
      } else if (this.queuedSeconds <= 0 && this.nextStart - this.ctx.currentTime < 0.15) {
        this.fadeTo(0)
        this.playing = false
        this.need = this.reservoir.rebufferSeconds
        this.onStarved()
      }
      if (this.playing) this.pump()
    }
    this.onReport({ playing: this.playing, buffered, need: this.need })
  }

  private pump() {
    while (this.queue.length > 0 && this.nextStart - this.ctx.currentTime < 0.5) {
      const buffer = this.queue.shift() as AudioBuffer
      this.queuedSeconds -= buffer.duration
      const source = this.ctx.createBufferSource()
      source.buffer = buffer
      source.connect(this.gate)
      if (this.nextStart < this.ctx.currentTime + 0.02) {
        this.nextStart = this.ctx.currentTime + 0.02
      }
      source.start(this.nextStart)
      this.nextStart += buffer.duration
      this.scheduled.push(source)
      source.onended = () => {
        this.scheduled = this.scheduled.filter((node) => node !== source)
      }
    }
  }

  private fadeTo(value: number) {
    const now = this.ctx.currentTime
    this.gate.gain.cancelScheduledValues(now)
    this.gate.gain.setValueAtTime(this.gate.gain.value, now)
    this.gate.gain.linearRampToValueAtTime(value, now + 0.08)
  }

  private stopScheduled() {
    for (const source of this.scheduled) {
      try {
        source.stop()
      } catch {
        // already finished
      }
    }
    this.scheduled = []
    this.nextStart = 0
  }
}

export class MrtStream {
  private ws: WebSocket | null = null
  private ctx: AudioContext | null = null
  private sink: PcmSink | null = null
  private sinkReady: Promise<PcmSink> | null = null
  private gain: GainNode | null = null
  private analyser: AnalyserNode | null = null
  private levelBuffer: Uint8Array = new Uint8Array(0)
  private freqBuffer: Uint8Array = new Uint8Array(0)

  private sampleRate = 48000
  private channels = 2

  private volume = 1
  private mood = "neutral"
  private instrument = "guitar"
  private wantsAudio = false
  private backendActive = false
  private realtimeFactor = 0
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null
  private suspendTimer: ReturnType<typeof setTimeout> | null = null
  private reconnectDelay = 500
  private lastPressureAt = 0
  private pendingPcm: ArrayBuffer[] = []
  private destroyed = false
  private sinkGeneration = 0
  private destroyPromise: Promise<void> | null = null

  private state: StreamState = {
    status: "idle",
    queuePosition: 0,
    listeners: 0,
    capacity: 0,
    message: null,
    bufferProgress: 0,
  }

  constructor(private onState: (state: StreamState) => void) {}

  // --- public api ---

  async start(mood: string, instrument: string) {
    if (this.destroyed) return
    this.mood = mood
    this.instrument = instrument
    this.wantsAudio = true
    if (this.suspendTimer) {
      clearTimeout(this.suspendTimer)
      this.suspendTimer = null
    }

    const ctx = this.ensureContext()
    // browsers start the context suspended until a user gesture; this call is
    // inside the click handler, so it is allowed to resume. it goes before the
    // await so the gesture is still on the stack when it lands.
    const resumed = ctx.resume()

    if (
      this.ws &&
      (this.ws.readyState === WebSocket.OPEN ||
        this.ws.readyState === WebSocket.CONNECTING)
    ) {
      // already have a socket - unpause on it rather than opening a second one.
      // while still connecting this is a no-op, and the hello we send on open
      // does the same job.
      this.send({ type: "resume" })
    } else {
      // WebSocket/model startup and AudioWorklet fetch+compile are independent;
      // overlap them so cold-start latency is their maximum, not their sum.
      this.connect()
    }

    const sink = await this.currentSink()
    await resumed
    // the module load is a round trip, and pause is one click away
    if (!sink || this.destroyed || !this.wantsAudio) return
    sink.play(true)
  }

  pause() {
    if (this.destroyed) return
    this.wantsAudio = false
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
    if (this.ws?.readyState === WebSocket.CONNECTING) this.ws.close()
    else this.send({ type: "pause" })
    this.sink?.stop()
    // Let the audio-thread fade complete, then stop consuming CPU (and stop
    // TimerSink sources from advancing silently) for the rest of the pause.
    if (this.ctx) {
      const ctx = this.ctx
      this.suspendTimer = setTimeout(() => {
        this.suspendTimer = null
        if (!this.wantsAudio && ctx.state !== "closed") void ctx.suspend()
      }, 80)
    }
    this.patch({ status: "paused", bufferProgress: 0 })
  }

  setStyle(mood: string, instrument: string) {
    if (this.destroyed) return
    this.mood = mood
    this.instrument = instrument
    this.send({ type: "style", mood, instrument })
  }

  setVolume(value: number) {
    if (this.destroyed) return
    this.volume = value
    if (this.gain && this.ctx) {
      // a short ramp instead of a step, so dragging the slider doesn't click
      this.gain.gain.setTargetAtTime(value, this.ctx.currentTime, 0.02)
    }
  }

  // instantaneous output level in 0..1, for the visualiser
  level(): number {
    if (!this.analyser) return 0
    this.analyser.getByteTimeDomainData(this.levelBuffer)

    let sum = 0
    for (let i = 0; i < this.levelBuffer.length; i++) {
      const centred = (this.levelBuffer[i] - 128) / 128
      sum += centred * centred
    }
    return Math.sqrt(sum / this.levelBuffer.length)
  }

  // frequency magnitudes in 0..255, for the visualiser. returns the number of
  // bins written, or 0 if there is no audio graph yet.
  spectrum(out: Uint8Array): number {
    if (!this.analyser) return 0
    const bins = Math.min(out.length, this.analyser.frequencyBinCount)
    if (this.freqBuffer.length !== this.analyser.frequencyBinCount) {
      this.freqBuffer = new Uint8Array(this.analyser.frequencyBinCount)
    }
    this.analyser.getByteFrequencyData(this.freqBuffer)
    out.set(this.freqBuffer.subarray(0, bins))
    return bins
  }

  destroy(): Promise<void> {
    if (this.destroyPromise) return this.destroyPromise
    this.destroyed = true
    this.wantsAudio = false
    this.backendActive = false
    this.sinkGeneration += 1

    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
    if (this.suspendTimer) {
      clearTimeout(this.suspendTimer)
      this.suspendTimer = null
    }

    const sink = this.sink
    const pendingSink = this.sinkReady
    this.sink = null
    this.sinkReady = null
    sink?.dispose()

    const ws = this.ws
    this.ws = null
    if (ws) {
      ws.onopen = null
      ws.onmessage = null
      ws.onerror = null
      ws.onclose = null
      ws.close()
    }

    this.pendingPcm = []
    this.gain?.disconnect()
    this.analyser?.disconnect()
    this.gain = null
    this.analyser = null
    this.levelBuffer = new Uint8Array(0)
    this.freqBuffer = new Uint8Array(0)

    const ctx = this.ctx
    this.ctx = null
    const closeContext = ctx?.state === "closed" ? Promise.resolve() : ctx?.close()
    const disposeLateSink = pendingSink
      ?.then((lateSink) => {
        if (lateSink !== sink) lateSink.dispose()
      })
      .catch(() => {})

    this.destroyPromise = Promise.all([closeContext, disposeLateSink]).then(() => {})
    return this.destroyPromise
  }

  // --- audio graph ---

  private ensureContext(): AudioContext {
    if (this.destroyed) throw new SinkBuildCancelled("stream was destroyed")
    if (this.ctx) return this.ctx

    // ask for 48k to match the model; if the device refuses, the worklet's read
    // cursor resamples for free on its way out
    const ctx = new AudioContext({ sampleRate: this.sampleRate })

    const gain = ctx.createGain()
    gain.gain.value = this.volume

    const analyser = ctx.createAnalyser()
    analyser.fftSize = 2048
    // the visualiser draws one dot column per bin group; without smoothing the
    // ring flickers a whole ring-step between frames
    analyser.smoothingTimeConstant = 0.75

    gain.connect(analyser)
    analyser.connect(ctx.destination)

    this.ctx = ctx
    this.gain = gain
    this.analyser = analyser
    this.levelBuffer = new Uint8Array(analyser.fftSize)
    this.freqBuffer = new Uint8Array(analyser.frequencyBinCount)
    return ctx
  }

  private ensureSink(): Promise<PcmSink> {
    if (this.sinkReady) return this.sinkReady
    const generation = this.sinkGeneration
    const ready = this.buildSink(generation)
    this.sinkReady = ready
    void ready.catch(() => {
      if (this.sinkReady === ready) this.sinkReady = null
    })
    return ready
  }

  private async currentSink(): Promise<PcmSink | null> {
    while (!this.destroyed) {
      try {
        return await this.ensureSink()
      } catch (error) {
        if (!(error instanceof SinkBuildCancelled)) throw error
      }
    }
    return null
  }

  private async buildSink(generation: number): Promise<PcmSink> {
    const ctx = this.ensureContext()
    const reservoir = reservoirFor(this.realtimeFactor)
    const isCurrent = () =>
      !this.destroyed && generation === this.sinkGeneration && this.ctx === ctx

    let sink: PcmSink
    try {
      if (!ctx.audioWorklet) throw new Error("no AudioWorklet")
      await ctx.audioWorklet.addModule(WORKLET_URL)
      if (!isCurrent()) throw new SinkBuildCancelled("sink build was superseded")
      const node = new AudioWorkletNode(ctx, "mrt-pcm", {
        numberOfInputs: 0,
        numberOfOutputs: 1,
        outputChannelCount: [this.channels],
        processorOptions: {
          channels: this.channels,
          sourceRate: this.sampleRate,
          ringSeconds: RING_SECONDS,
          ...reservoir,
        },
      })
      sink = new WorkletSink(
        node,
        (report) => this.onSinkReport(report),
        () => this.onStarved(),
      )
    } catch (error) {
      if (!isCurrent() || error instanceof SinkBuildCancelled) {
        throw new SinkBuildCancelled("sink build was superseded")
      }
      // no worklet here; chunk scheduling still makes sound
      sink = new TimerSink(
        ctx,
        this.channels,
        this.sampleRate,
        (report) => this.onSinkReport(report),
        () => this.onStarved(),
      )
      sink.configure(reservoir)
    }

    if (!isCurrent() || !this.gain) {
      sink.dispose()
      throw new SinkBuildCancelled("sink build was superseded")
    }
    sink.output.connect(this.gain)
    // Status may have arrived while the worklet module was compiling.
    sink.configure(reservoirFor(this.realtimeFactor))
    this.sink = sink
    for (const pcm of this.pendingPcm.splice(0)) sink.push(pcm)
    return sink
  }

  private async rebuildSink() {
    if (this.destroyed) return
    this.sinkGeneration += 1
    const oldSink = this.sink
    const oldReady = this.sinkReady
    oldSink?.dispose()
    this.sink = null
    this.sinkReady = null
    void oldReady
      ?.then((lateSink) => {
        if (lateSink !== oldSink) lateSink.dispose()
      })
      .catch(() => {})
    const sink = await this.currentSink()
    if (sink && !this.destroyed && this.wantsAudio) sink.play(true)
  }

  private onSinkReport(report: SinkReport) {
    if (!this.wantsAudio) return
    if (!this.backendActive) return
    // Let the tuner recover while audio remains instead of waiting until the
    // listener hears a gap. Reports are frequent, so match the server's dwell.
    if (report.playing && report.buffered < 0.75) {
      const now = performance.now()
      if (now - this.lastPressureAt >= 6000) {
        this.lastPressureAt = now
        this.send({ type: "pressure" })
      }
    }
    if (report.playing) {
      this.patch({ status: "live", bufferProgress: 1 })
    } else {
      this.patch({
        status: "buffering",
        bufferProgress: report.need > 0 ? Math.min(1, report.buffered / report.need) : 0,
      })
    }
  }

  private onStarved() {
    // only the client knows a gap was audible, so it is the one that reports it
    this.send({ type: "gap" })
  }

  private enqueue(pcm: ArrayBuffer) {
    const sink = this.sink
    if (!sink) {
      // Normally the worklet is ready before the model's first second. Keep a
      // small bound for cold caches instead of dropping the beginning.
      if (this.pendingPcm.length >= 4) this.pendingPcm.shift()
      this.pendingPcm.push(pcm)
      return
    }

    // Also accept the final in-flight chunk after pause. The worklet retains
    // it silently, preserving sample continuity when playback resumes.
    sink.push(pcm)
  }

  // --- websocket ---

  private connect() {
    if (this.destroyed) return
    if (
      this.ws &&
      (this.ws.readyState === WebSocket.OPEN ||
        this.ws.readyState === WebSocket.CONNECTING)
    ) {
      return
    }
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:"
    const ws = new WebSocket(`${protocol}//${backendHost()}/ws/session`)
    ws.binaryType = "arraybuffer"
    this.ws = ws
    this.patch({ status: "connecting", message: null })

    ws.onopen = () => {
      if (this.destroyed || this.ws !== ws) return
      this.reconnectDelay = 500
      this.send({
        type: "hello",
        sessionId: readSessionId(),
        mood: this.mood,
        instrument: this.instrument,
      })
      if (!this.wantsAudio) this.send({ type: "pause" })
    }

    ws.onmessage = (event) => {
      if (this.destroyed || this.ws !== ws) return
      if (event.data instanceof ArrayBuffer) {
        this.enqueue(event.data)
        return
      }
      this.handleControl(JSON.parse(event.data as string))
    }

    ws.onerror = () => {
      if (this.destroyed || this.ws !== ws) return
      this.patch({ status: "error", message: "lost the backend" })
    }

    ws.onclose = (event) => {
      if (this.destroyed || this.ws !== ws) return
      this.ws = null
      this.backendActive = false
      if (event.code === 1013) this.sink?.reset()
      this.pendingPcm = []
      if (TERMINAL_SERVER_CLOSE_CODES.has(event.code)) {
        this.patch({ status: "idle", message: "application stopped", bufferProgress: 0 })
        void this.destroy()
        return
      }
      if (!this.wantsAudio) {
        this.patch({ status: "idle", bufferProgress: 0 })
        return
      }
      this.patch({ status: "connecting", message: "reconnecting" })
      this.reconnectTimer = setTimeout(() => {
        this.reconnectTimer = null
        if (!this.destroyed) this.connect()
      }, this.reconnectDelay)
      this.reconnectDelay = Math.min(this.reconnectDelay * 2, 8000)
    }
  }

  private handleControl(message: Record<string, unknown>) {
    switch (message.type) {
      case "hello": {
        const id = message.sessionId as string
        writeSessionId(id)

        if (message.resumed === false && this.sink) {
          // The backend could not continue the prior state (or this is the
          // first session). Fade/reset before accepting an unrelated stream.
          this.sink.reset()
          if (this.wantsAudio) this.sink.play(true)
        }

        // the sink is built before this arrives, from the format we assume the
        // backend uses. it always has so far - but a mismatch would play at the
        // wrong pitch and say nothing about it, so rebuild rather than trust it
        const rate = (message.sampleRate as number) ?? this.sampleRate
        const channels = (message.channels as number) ?? this.channels
        const moved = rate !== this.sampleRate || channels !== this.channels
        this.sampleRate = rate
        this.channels = channels
        if (moved && this.sink) void this.rebuildSink()
        break
      }

      case "status": {
        const backendState = message.state as string
        const wasActive = this.backendActive
        this.backendActive = backendState === "active"

        const factor = (message.realtimeFactor as number) ?? 0
        if (factor > 0 && Math.abs(factor - this.realtimeFactor) > 0.05) {
          // the backend has told us how fast it really renders; size the
          // reservoir to match rather than making every machine pay for the
          // slowest one
          this.realtimeFactor = factor
          this.sink?.configure(reservoirFor(factor))
        }

        if (this.backendActive && !wasActive && this.wantsAudio) {
          // A freshly playing one-second bank naturally dips while the second
          // chunk renders. Give that sawtooth a full feedback interval before
          // treating low water as sustained pressure.
          this.lastPressureAt = performance.now()
          this.sink?.play(true)
        }

        let status: StreamStatus
        if (this.backendActive) {
          // the backend is generating for us; what the listener hears depends
          // on whether we have banked enough to play
          status = !this.wantsAudio
            ? "paused"
            : this.state.status === "live"
              ? "live"
              : "buffering"
        } else if (backendState === "queued") {
          status = "queued"
        } else if (backendState === "loading") {
          status = "loading"
        } else {
          status = "paused"
        }

        this.patch({
          status,
          queuePosition: (message.position as number) ?? 0,
          listeners: (message.listeners as number) ?? 0,
          capacity: (message.capacity as number) ?? 0,
          message: (message.error as string) ?? null,
        })
        break
      }

      case "error": {
        this.patch({ status: "error", message: message.message as string })
        break
      }
    }
  }

  private send(payload: Record<string, unknown>) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(payload))
    }
  }

  private patch(next: Partial<StreamState>) {
    if (this.destroyed) return
    // the sink reports about twelve times a second, and every one of these used
    // to re-render the whole page. once the stream is live nothing in here
    // moves, so an unchanged state is not worth waking react for.
    const merged = { ...this.state, ...next }
    merged.bufferProgress = Math.round(merged.bufferProgress * 50) / 50
    const current = this.state
    if (
      merged.status === current.status &&
      merged.queuePosition === current.queuePosition &&
      merged.listeners === current.listeners &&
      merged.capacity === current.capacity &&
      merged.message === current.message &&
      merged.bufferProgress === current.bufferProgress
    ) {
      return
    }
    this.state = merged
    this.onState(merged)
  }
}
