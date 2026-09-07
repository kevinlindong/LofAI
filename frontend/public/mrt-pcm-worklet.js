// ring-buffer pcm player, running on the audio thread
//
// the backend hands us interleaved int16 chunks whenever it has them, which is
// nothing like a steady clock. the old client answered that by scheduling one
// AudioBufferSourceNode per chunk from a setInterval, which has two problems:
// the timer lives on the main thread, so a busy tab pushes the schedule late
// and the stream clicks, and each source resamples from its own phase, so any
// playback rate other than 1.0 puts a discontinuity at every chunk seam - a
// tick two or three times a second, which is what "distorted" sounded like.
//
// this processor owns a ring buffer and one fractional read cursor instead.
// everything is a single continuous signal: chunk seams do not exist, the
// playback rate can move anywhere without a seam to click at, and none of it
// touches the main thread, so no amount of canvas work can stutter the music.

const REPORT_EVERY = 32 // render quanta between state messages (~85ms)

class PcmProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super()
    const opts = options.processorOptions || {}

    this.channels = opts.channels || 2
    // the source rate, which is not always the context rate: a device that
    // refuses 48k gets resampled by the read cursor for free
    this.srcRate = opts.sourceRate || sampleRate
    this.baseStep = this.srcRate / sampleRate

    this.capacity = Math.max(8192, Math.round((opts.ringSeconds || 8) * this.srcRate))
    // PCM arrives as int16. Keeping that representation removes a full-buffer
    // conversion on the UI thread and halves the ring's memory footprint.
    this.ring = new Int16Array(this.capacity * this.channels)

    // absolute source-frame counters. floats, but exact well past any session
    // length: 2^53 frames at 48khz is about 6000 years.
    this.written = 0
    this.readPos = 0

    this.prebuffer = Math.round((opts.prebufferSeconds ?? 0.8) * this.srcRate)
    this.rebuffer = Math.round((opts.rebufferSeconds ?? 0.8) * this.srcRate)
    this.comfort = Math.round((opts.comfortSeconds ?? 1.6) * this.srcRate)
    this.minRate = opts.minRate ?? 1
    this.need = this.prebuffer
    this.fresh = true

    this.want = false
    this.playing = false
    this.stopping = false
    this.resetting = false
    this.gain = 0
    this.gainTarget = 0
    this.fadeSeconds = Math.max(0.004, opts.fadeSeconds ?? 0.02)
    this.fadeStep = 1 / (this.fadeSeconds * sampleRate)

    this.rate = 1
    this.sinceReport = 0
    this.gaps = 0
    this.overflows = 0
    this.pendingPcm = []
    this.pendingFrames = 0
    this.disposed = false

    this.port.onmessage = (event) => this.receive(event.data)
  }

  receive(msg) {
    if (this.disposed) return
    switch (msg.type) {
      case "pcm":
        this.write(new Int16Array(msg.pcm))
        break

      case "config":
        if (msg.prebufferSeconds != null)
          this.prebuffer = Math.round(msg.prebufferSeconds * this.srcRate)
        if (msg.rebufferSeconds != null)
          this.rebuffer = Math.round(msg.rebufferSeconds * this.srcRate)
        if (msg.comfortSeconds != null)
          this.comfort = Math.round(msg.comfortSeconds * this.srcRate)
        if (msg.minRate != null) this.minRate = msg.minRate
        if (!this.playing) this.need = this.fresh ? this.prebuffer : this.rebuffer
        break

      case "play":
        // start (or resume) as soon as the reservoir is deep enough
        this.want = true
        if (this.resetting) break
        this.stopping = false
        if (!this.playing) {
          this.fresh = msg.fresh
          this.need = this.fresh ? this.prebuffer : this.rebuffer
        }
        break

      case "stop":
        // fade out but keep the reservoir, so an unpause picks up where it left
        this.want = false
        this.stopping = true
        break

      case "reset":
        // A new stream cannot splice into the old one. Fade the old reservoir
        // first; clearing an arbitrary sample immediately is an audible click.
        this.beginReset()
        break

      case "dispose":
        this.disposed = true
        this.playing = false
        this.stopping = true
        this.gain = 0
        this.gainTarget = 0
        this.written = 0
        this.readPos = 0
        this.ring = new Int16Array(0)
        this.pendingPcm = []
        this.pendingFrames = 0
        this.port.onmessage = null
        if (typeof this.port.close === "function") this.port.close()
        break
    }
  }

  beginReset() {
    this.pendingPcm = []
    this.pendingFrames = 0
    if (this.playing || this.gain > 0) {
      this.resetting = true
      this.stopping = true
    } else {
      this.finishReset()
    }
  }

  finishReset() {
    const pending = this.pendingPcm
    this.pendingPcm = []
    this.pendingFrames = 0
    this.written = 0
    this.readPos = 0
    this.playing = false
    this.stopping = !this.want
    this.resetting = false
    this.gain = 0
    this.gainTarget = 0
    this.rate = 1
    this.need = this.prebuffer
    this.fresh = true
    for (const samples of pending) this.write(samples)
  }

  queueAfterReset(samples) {
    const frames = samples.length / this.channels
    if (this.pendingFrames + frames > this.capacity) {
      // A suspended audio thread can receive several seconds in one batch.
      // Keep one contiguous new segment, bounded by the ring's capacity.
      this.pendingPcm = []
      this.pendingFrames = 0
    }
    this.pendingPcm.push(samples)
    this.pendingFrames += frames
  }

  write(samples) {
    const channels = this.channels
    let frames = Math.floor(samples.length / channels)
    if (frames <= 0) return
    // Only whole frames fit in the ring. Oversized packets retain a bounded,
    // contiguous tail instead of allocating an unbounded reset queue.
    if (frames > this.capacity) {
      samples = samples.slice((frames - this.capacity) * channels, frames * channels)
      frames = this.capacity
    } else if (samples.length !== frames * channels) {
      samples = samples.subarray(0, frames * channels)
    }
    if (this.resetting) {
      this.queueAfterReset(samples)
      return
    }

    // Dropping just this packet would splice the next accepted packet onto
    // unrelated samples later. Fade to a fresh contiguous segment instead.
    const unread = this.written - Math.floor(this.readPos)
    if (frames > this.capacity - unread) {
      this.overflows += 1
      this.beginReset()
      if (this.resetting) {
        this.queueAfterReset(samples)
        return
      }
    }

    const start = this.written % this.capacity
    const first = Math.min(frames, this.capacity - start)
    this.ring.set(samples.subarray(0, first * channels), start * channels)
    if (first < frames) {
      this.ring.set(samples.subarray(first * channels), 0)
    }
    this.written += frames
  }

  process(_inputs, outputs) {
    if (this.disposed) return false
    const out = outputs[0]
    if (!out || out.length === 0) return true

    const blockSize = out[0].length
    const outChannels = out.length
    const channels = this.channels
    const available = this.written - this.readPos

    // source frames this block will consume, plus the one extra the
    // interpolator reads ahead
    const consumes = blockSize * this.rate * this.baseStep + 2
    // start fading before the reservoir is actually dry, so silence is always
    // reached by a ramp and never by a cliff. half again as much as the fade
    // needs, because the fade itself is still drinking while it runs.
    const lowWater = consumes + this.fadeSeconds * this.srcRate * 1.5

    if (this.playing) {
      if (this.stopping || available < lowWater) {
        this.gainTarget = 0
      } else {
        this.gainTarget = 1
      }
      if (this.gainTarget === 0 && this.gain <= 0) {
        this.playing = false
        if (this.resetting) {
          this.finishReset()
        } else if (!this.stopping) {
          // ran dry on its own: the listener heard a gap
          this.gaps += 1
          this.fresh = false
          this.need = this.rebuffer
          this.port.postMessage({ type: "starved" })
        }
      }
    } else if (this.want && !this.stopping && available >= this.need) {
      this.playing = true
      this.gainTarget = 1
    }

    if (!this.playing && this.gain <= 0) {
      for (let c = 0; c < outChannels; c++) out[c].fill(0)
      this.report(this.written - this.readPos)
      return true
    }

    this.rate = this.nextRate(available)

    const ring = this.ring
    const capacity = this.capacity
    const step = this.rate * this.baseStep
    const fadeStep = this.fadeStep
    const target = this.gainTarget
    // The normal 48 kHz, unity-rate path is an exact int16-to-float copy.
    // It needs neither interpolation nor a second ring read per sample.
    const nativeRate = step === 1 && Number.isInteger(this.readPos)
    const limit = this.written - (nativeRate ? 0 : 1)
    const copyChannels = Math.min(channels, outChannels)

    let pos = this.readPos
    let ringFrame = Math.floor(pos) % capacity
    let gain = this.gain

    for (let i = 0; i < blockSize; i++) {
      if (pos >= limit) {
        // hard floor: the pre-emptive fade should mean we are already silent
        for (let c = 0; c < outChannels; c++) out[c][i] = 0
        gain = 0
        continue
      }

      if (nativeRate) {
        const a = ringFrame * channels
        for (let c = 0; c < copyChannels; c++) out[c][i] = (ring[a + c] / 32768) * gain
        ringFrame += 1
        if (ringFrame === capacity) ringFrame = 0
      } else {
        const index = Math.floor(pos)
        const frac = pos - index
        const a = (index % capacity) * channels
        const b = ((index + 1) % capacity) * channels
        for (let c = 0; c < copyChannels; c++) {
          const s0 = ring[a + c]
          out[c][i] = ((s0 + (ring[b + c] - s0) * frac) / 32768) * gain
        }
      }
      for (let c = copyChannels; c < outChannels; c++) {
        out[c][i] = out[0][i]
      }

      if (gain < target) gain = gain + fadeStep > target ? target : gain + fadeStep
      else if (gain > target) gain = gain - fadeStep < target ? target : gain - fadeStep

      pos += step
    }

    this.readPos = pos
    this.gain = gain
    this.report(this.written - pos)
    return true
  }

  // Ease toward a configured fallback rate as the reservoir runs low. The
  // production stream pins minRate to 1 so melody pitch never wanders; the
  // mechanism remains for hosts that explicitly choose time stretch.
  nextRate(available) {
    if (!this.playing || this.minRate === 1) return 1
    const t = Math.min(1, Math.max(0, available / this.comfort))
    const target = this.minRate + (1 - this.minRate) * t
    // one step per render quantum: the signal stays continuous either way,
    // only its rate of change is quantised, which is inaudible
    return this.rate + (target - this.rate) * 0.02
  }

  report(available) {
    if (this.disposed) return
    this.sinceReport += 1
    if (this.sinceReport < REPORT_EVERY) return
    this.sinceReport = 0
    this.port.postMessage({
      type: "state",
      playing: this.playing && this.gain > 0,
      buffered: available / this.srcRate,
      need: this.need / this.srcRate,
      rate: this.rate,
      gaps: this.gaps,
      overflows: this.overflows,
    })
  }
}

registerProcessor("mrt-pcm", PcmProcessor)
