"use client"

import { useEffect, useRef, type ReactNode } from "react"
import { BlobSet } from "@/lib/dot-field"
import {
  buildWave,
  SPOKES,
  SURFACE,
  WAVE_BODY,
  WAVE_CREST,
  WAVE_GLOW,
  WAVE_GLOW_AT,
  WAVE_LIP,
} from "@/lib/wave-scene"

interface DotVisualizerProps {
  // fills `out` with frequency magnitudes and returns how many bins it wrote
  getSpectrum: (out: Uint8Array) => number
  active: boolean
  children?: ReactNode
}

// the polar dot matrix the wave is struck on. the dots sit at a constant pitch
// rather than a constant angle - each ring gets as many as fit around it - so
// the field stays even instead of fanning into rays at the edge. what lights
// them is a ring of metaballs; that part lives in lib/wave-scene.
const RINGS = 11
const INNER = 0.52
const OUTER = 0.96

// how quickly a column follows the spectrum, as time constants in seconds:
// snaps up to a transient, settles back slowly. these are seconds rather than
// per-frame fractions so the wave moves the same at any frame rate.
const ATTACK = 0.022
const RELEASE = 0.1

// each band is read against its own recent range rather than against the raw
// 0..255 the fft hands back, because those numbers are not a level - they are
// a level plus whatever the track's mix, the master gain and the codec left
// in that part of the spectrum. read raw, the top bands sit near the floor all
// night and the bass sits near the ceiling, so the ring has one shape and only
// breathes. read against a range that follows the band, every band gets the
// whole radius to move in and the panel finally shows the music rather than
// the mix. the ceiling drops slowly so a loud bar keeps its scale for a few
// seconds; the floor rises slower still, so a band that goes quiet takes its
// time admitting it. both are per second.
const CEIL_FALL = 0.14
const FLOOR_RISE = 0.05

// the smallest range a band is allowed to be scaled against, so a band that
// is doing nothing at all is not amplified into pure noise
const MIN_RANGE = 0.1

// every frame the display offers. the wave's edge crawls a fraction of a dot
// at a time, and at thirty that crawl reads as a stutter. sampling the field
// over the whole lattice costs a tenth of a millisecond.
const MIN_FRAME_MS = 0

const TAU = Math.PI * 2

const PALETTE_VARS = ["--dot-0", "--dot-1", "--dot-2", "--dot-3", "--dot-4", "--dot-5", "--dot-6"]

export function DotVisualizer({ getSpectrum, active, children }: DotVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const wrapRef = useRef<HTMLDivElement>(null)
  const activeRef = useRef(active)
  const specRef = useRef(getSpectrum)
  activeRef.current = active
  specRef.current = getSpectrum

  useEffect(() => {
    const canvas = canvasRef.current
    const wrap = wrapRef.current
    if (!canvas || !wrap) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // set only when the animation loop is not running, so the single drawn
    // frame can be re-struck after a resize or a theme change
    let redrawStill: (() => void) | null = null

    let palette = PALETTE_VARS.map(() => "#000")
    const readPalette = () => {
      const style = getComputedStyle(document.documentElement)
      palette = PALETTE_VARS.map((v) => style.getPropertyValue(v).trim() || "#000")
    }
    readPalette()
    const themeWatch = new MutationObserver(() => {
      readPalette()
      redrawStill?.()
    })
    themeWatch.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] })

    let size = 0
    let dpr = 0
    // assigning canvas.width wipes the canvas, so only assign when something
    // actually changed - otherwise the observer's first callback erases the
    // frame that has just been drawn
    const resize = (): boolean => {
      const nextDpr = Math.min(2, window.devicePixelRatio || 1)
      const nextSize = Math.max(120, Math.min(wrap.clientWidth, wrap.clientHeight))
      if (nextSize === size && nextDpr === dpr) return false
      size = nextSize
      dpr = nextDpr
      canvas.width = size * dpr
      canvas.height = size * dpr
      canvas.style.width = `${size}px`
      canvas.style.height = `${size}px`
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      return true
    }
    resize()
    const sizeWatch = new ResizeObserver(() => {
      if (!resize()) return
      layout()
      redrawStill?.()
    })
    sizeWatch.observe(wrap)

    // the fft bins each spoke of the half-wave covers, spaced quadratically so
    // the bass does not swallow the first three spokes and leave the rest
    // flat. a spoke takes the mean of its whole band rather than one bin out
    // of the sixty-odd it spans: up at the top a single bin is as likely to
    // land in a null between partials as on the note, which is a spoke that
    // twitches on nothing.
    const half = SPOKES / 2
    const bins = new Uint8Array(1024)
    const bandLo = new Int32Array(half)
    const bandHi = new Int32Array(half)
    for (let i = 0; i < half; i++) {
      bandLo[i] = Math.round(2 + Math.pow(i / half, 2) * 338)
      bandHi[i] = Math.round(2 + Math.pow((i + 1) / half, 2) * 338)
    }

    // the range each band is currently being read against, and the same for
    // the ring as a whole
    const bandFloor = new Float32Array(half)
    const bandCeil = new Float32Array(half)
    bandFloor.fill(1)
    let loudCeil = 0

    const amp = new Float32Array(SPOKES)
    const shape = new Float32Array(half)

    // the dot lattice, rebuilt only when the canvas changes size. deriving it
    // inside the draw loop meant a thousand-odd sin/cos pairs a frame; the
    // field is what moves now, and the dots it is sampled at never do.
    let dotX = new Float32Array(0)
    let dotY = new Float32Array(0)
    // for each dot, the dot one ring further out and one ring further in, or
    // -1 off the ends of the lattice. the crest is taken from these rather
    // than from a band of field values: the field falls off at a rate that
    // depends on how big the blobs currently are, so a rim cut by value is two
    // rings thick on a loud bar and gone on a quiet one, and every ring in
    // between lands on its own rung of the ramp - which reads as speckle. cut
    // from the neighbours it is exactly one dot wherever the surface happens
    // to be.
    let outward = new Int32Array(0)
    let inward = new Int32Array(0)
    let value = new Float32Array(0)
    let dotRadius = 1
    let inner = 0
    let pitch = 1
    let shades: Float32Array[] = []
    const counts = new Int32Array(PALETTE_VARS.length)
    const blobs = new BlobSet()

    const layout = () => {
      const mid = size / 2
      inner = mid * INNER
      pitch = (mid * OUTER - inner) / (RINGS - 1)
      dotRadius = Math.max(1, pitch * 0.3)

      // as many dots as fit around each ring at the radial pitch
      const around = (r: number) =>
        Math.max(12, Math.round((TAU * (inner + r * pitch)) / pitch))

      const counted: number[] = []
      const starts: number[] = []
      let total = 0
      for (let r = 0; r < RINGS; r++) {
        starts.push(total)
        counted.push(around(r))
        total += counted[r]
      }

      dotX = new Float32Array(total)
      dotY = new Float32Array(total)
      outward = new Int32Array(total)
      inward = new Int32Array(total)
      value = new Float32Array(total)

      for (let r = 0; r < RINGS; r++) {
        const d = inner + r * pitch
        const n = counted[r]
        for (let k = 0; k < n; k++) {
          const i = starts[r] + k
          const a = (k / n) * TAU
          dotX[i] = Math.cos(a) * d
          dotY[i] = Math.sin(a) * d
          // the rings hold different numbers of dots, so a neighbour is
          // whichever dot of the next ring round sits closest in angle
          outward[i] =
            r + 1 < RINGS
              ? starts[r + 1] + (Math.round((k / n) * counted[r + 1]) % counted[r + 1])
              : -1
          inward[i] =
            r > 0 ? starts[r - 1] + (Math.round((k / n) * counted[r - 1]) % counted[r - 1]) : -1
        }
      }
      shades = PALETTE_VARS.map(() => new Float32Array(total * 2))
    }
    layout()

    let raf = 0
    let last = performance.now()

    const draw = (now: number, dt: number) => {
      const written = activeRef.current ? specRef.current(bins) : 0

      // one time constant per direction, not per spoke
      const rise = 1 - Math.exp(-dt / ATTACK)
      const fall = 1 - Math.exp(-dt / RELEASE)

      // how far up its own range each band is sitting, and how loud the ring
      // is against its own. shape is what the spoke draws; gain is what says
      // whether there is any music behind it at all, so that a rest between
      // phrases collapses the ring instead of leaving the spectrum's shape
      // hanging there at full scale.
      let gain = 0
      if (written > 0) {
        let loud = 0
        for (let i = 0; i < half; i++) {
          const hi = Math.max(1, Math.min(written, bandHi[i]))
          const lo = Math.min(bandLo[i], hi - 1)
          let sum = 0
          for (let b = lo; b < hi; b++) sum += bins[b]
          const lvl = sum / (hi - lo) / 255

          bandCeil[i] = Math.max(lvl, bandCeil[i] - dt * CEIL_FALL)
          bandFloor[i] = Math.min(lvl, bandFloor[i] + dt * FLOOR_RISE)
          const n = (lvl - bandFloor[i]) / Math.max(MIN_RANGE, bandCeil[i] - bandFloor[i])
          // a smoothstep on the way out: it pulls the quiet half of the range
          // down towards the lip and the loud half up towards the edge, so a
          // band crossing the middle crosses it as a lunge rather than a drift
          shape[i] = n * n * (3 - 2 * n)
          loud += lvl
        }
        loud /= half
        loudCeil = Math.max(loud, loudCeil - dt * CEIL_FALL)
        gain = loudCeil > 0.02 ? Math.min(1, loud / loudCeil) : 0
      }

      for (let s = 0; s < SPOKES; s++) {
        // mirror the half-wave across the ring so it reads as one continuous
        // shape instead of a spectrum that jumps at the seam
        const i = s < half ? s : SPOKES - 1 - s
        let target: number
        if (written > 0) {
          target = (0.18 + 0.82 * gain) * (0.05 + 0.95 * shape[i])
        } else {
          // at rest, a slow breath. it used to be a swell travelling round the
          // ring, which is a spin by another name - this one is the same on
          // every spoke, so nothing on the panel goes round.
          target = 0.1 + 0.045 * Math.sin(now / 1300) + 0.03 * Math.sin(i * 1.7)
        }
        amp[s] += (target - amp[s]) * (target > amp[s] ? rise : fall)
      }

      buildWave(blobs, amp, {
        inner,
        span: inner * (OUTER / INNER - 1),
        pitch,
      })

      ctx.clearRect(0, 0, size, size)
      counts.fill(0)

      for (let i = 0; i < value.length; i++) value[i] = blobs.at(dotX[i], dotY[i])

      for (let i = 0; i < value.length; i++) {
        const v = value[i]
        let shade: number
        if (v >= SURFACE) {
          const out = outward[i]
          const inn = inward[i]
          // off either end of the lattice counts as empty, so a wave that has
          // run past the edge still gets a line drawn along it
          shade =
            out < 0 || value[out] < SURFACE
              ? WAVE_CREST
              : inn < 0 || value[inn] < SURFACE
                ? WAVE_LIP
                : WAVE_BODY
        } else if (v >= WAVE_GLOW_AT) {
          shade = WAVE_GLOW
        } else {
          continue
        }
        const at = counts[shade]
        const buffer = shades[shade]
        buffer[at] = dotX[i]
        buffer[at + 1] = dotY[i]
        counts[shade] = at + 2
      }

      const mid = size / 2
      ctx.save()
      ctx.translate(mid, mid)
      for (let v = 0; v < shades.length; v++) {
        const filled = counts[v]
        if (filled === 0) continue
        const buffer = shades[v]
        ctx.fillStyle = palette[v]
        ctx.beginPath()
        for (let i = 0; i < filled; i += 2) {
          ctx.moveTo(buffer[i] + dotRadius, buffer[i + 1])
          ctx.arc(buffer[i], buffer[i + 1], dotRadius, 0, TAU)
        }
        ctx.fill()
      }
      ctx.restore()
    }

    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      // a whole second of dt, so the one frame we draw is the settled one the
      // animation would have eased to rather than a single step towards it
      redrawStill = () => draw(performance.now(), 1)
      redrawStill()
      return () => {
        themeWatch.disconnect()
        sizeWatch.disconnect()
      }
    }

    const tick = (now: number) => {
      raf = requestAnimationFrame(tick)
      const since = now - last
      if (since < MIN_FRAME_MS) return
      last = now
      draw(now, Math.min(0.1, since / 1000))
    }
    raf = requestAnimationFrame(tick)

    return () => {
      cancelAnimationFrame(raf)
      themeWatch.disconnect()
      sizeWatch.disconnect()
    }
  }, [])

  return (
    <div ref={wrapRef} className="relative aspect-square w-full max-w-[min(52vh,29rem)]">
      <canvas ref={canvasRef} className="absolute inset-0 m-auto" aria-hidden />
      <div className="absolute inset-0 flex items-center justify-center">{children}</div>
    </div>
  )
}

export default DotVisualizer
