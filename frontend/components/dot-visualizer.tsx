"use client"

import { useEffect, useRef, type ReactNode } from "react"
import { BlobSet } from "@/lib/dot-field"
import { paintInk, type InkCells, type InkGeometry } from "@/lib/ink-render"
import {
  buildWave,
  SPOKES,
  SURFACE,
  WAVE_BODY,
  WAVE_CREST,
  WAVE_GLOW,
  WAVE_GLOW_AT,
} from "@/lib/wave-scene"

interface DotVisualizerProps {
  // fills `out` with frequency magnitudes and returns how many bins it wrote
  getSpectrum: (out: Uint8Array) => number
  active: boolean
  children?: ReactNode
}

// the dot matrix the wave is struck on. the dots sit in rank and file - the
// same square lattice as every other panel on the page - and the round wave
// lights whichever of them it reaches, which is what a circle is on a real
// matrix: a circle's worth of square cells. RINGS sets the pitch by dividing
// the wave's radial travel into that many rows; INNER and OUTER are where
// the wave's lip and its full-scale crest sit, and the lattice keeps a round
// hole clear where the transport button lives. the ring of metaballs doing
// the lighting lives in lib/wave-scene.
//
// the cells are not dots any more, they are ink. each one's field value is a
// continuous fill rather than a yes/no, so a cell the wave is only just
// reaching is a small bead and a cell inside a loud bar is saturated and
// overlapping its neighbours - and wherever two adjacent cells are both wet,
// lib/ink-render draws the pinched neck between them. the ring's edge beads,
// joins and flows instead of switching on a dot at a time.
const RINGS = 11
const INNER = 0.52
const OUTER = 0.96

// the cells are dots again, and what makes them ink is what happens between
// them: a dot that grows until it nears its neighbour pulls a tangent-continuous
// membrane across to it, so the ring reads as one poured body with waists in it.
// the numbers are fractions of the pitch, scaled once the pitch is known.
const INK: InkGeometry = {
  // a cell the wave has only breathed on. below about 0.29 of the pitch a dot
  // cannot reach its neighbour at all, so the quietest ink is honestly separate
  // beads and joining up is something the music does
  minRadius: 0.16,
  // at half the pitch two neighbours are exactly tangent; a little past it they
  // overlap and a loud passage fuses into one sheet
  maxRadius: 0.54,
  spread: 0.5,
  handleSize: 2.4,
  reach: 2.5,
  dryRadius: 0,
  diagonals: true,
  // the first fifth of a cell's fill is spent growing the dot in from nothing,
  // so the wave's edge arrives as a swelling bead and leaves as a shrinking
  // one, instead of blinking on and off at minRadius as the surface crawls
  swellIn: 0.2,
}

// how the field maps to ink. a cell wets at the same value that used to earn
// it a breath of glow and saturates a little under twice the surface, so the
// crest is always the fullest ink on the panel.
const WET_AT = WAVE_GLOW_AT
const FULL_AT = SURFACE * 1.9

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
    // inside the draw loop meant a thousand-odd distance checks a frame; the
    // field is what moves now, and the dots it is sampled at never do.
    let dotX = new Float32Array(0)
    let dotY = new Float32Array(0)
    // for each dot, its radial neighbour: the lattice dot nearest to one
    // pitch further out along this dot's own radius, or -1 off the lattice. the crest is taken from these rather than
    // from a band of field values: the field falls off at a rate that
    // depends on how big the blobs currently are, so a rim cut by value is
    // two dots thick on a loud bar and gone on a quiet one. cut from the
    // radial neighbours it is exactly one dot wherever the surface happens
    // to be - even though the dots themselves now sit in rank and file.
    let outward = new Int32Array(0)
    let value = new Float32Array(0)
    // per cell: whether the field is over the surface there, and whether the
    // cell is on the mask's radial rim (1 lip, 2 crest, 0 neither)
    let interior = new Uint8Array(0)
    let edge = new Uint8Array(0)
    let inner = 0
    let pitch = 1
    let ink: InkCells | null = null
    let inkGeo: InkGeometry = { ...INK }
    // the overlays' geometry: same dots, no membranes (reach 0 means no pair
    // is ever close enough to hold one)
    let lineGeo: InkGeometry = { ...INK, reach: 0 }
    const blobs = new BlobSet()

    // 0 below `from`, 1 above `to`, eased between. how much ink a cell holds is
    // read off the field through this, so the wave's edge arrives as a swelling
    // bead rather than as a cell switching on.
    const ramp = (from: number, to: number, v: number) => {
      const t = Math.min(1, Math.max(0, (v - from) / (to - from)))
      return t * t * (3 - 2 * t)
    }

    const layout = () => {
      const mid = size / 2
      inner = mid * INNER
      pitch = (mid * OUTER - inner) / (RINGS - 1)
      inkGeo = {
        minRadius: INK.minRadius * pitch,
        maxRadius: INK.maxRadius * pitch,
        spread: INK.spread,
        handleSize: INK.handleSize,
        reach: INK.reach,
        dryRadius: 0,
        diagonals: INK.diagonals,
        swellIn: INK.swellIn,
      }
      lineGeo = { ...inkGeo, reach: 0 }

      // a square lattice over the whole panel, odd-counted so a rank runs
      // through dead centre and the grid is visibly aligned with the button.
      // dots are struck out of a round hole in the middle where the button
      // sits; everywhere else they exist and simply stay unlit until the
      // wave reaches them, so the ring's edges are honestly grid-quantised
      // rather than smoothed by a lattice bent to fit them.
      const n = (Math.floor(size / pitch) - 1) | 1
      const off = (n - 1) / 2
      const hole = inner - pitch * 0.55
      const rim = mid * OUTER + pitch * 0.75
      const map = new Int32Array(n * n).fill(-1)
      const xs: number[] = []
      const ys: number[] = []
      for (let gy = 0; gy < n; gy++) {
        for (let gx = 0; gx < n; gx++) {
          const x = (gx - off) * pitch
          const y = (gy - off) * pitch
          const d = Math.hypot(x, y)
          // inside the hole the button lives; past the rim the wave cannot
          // reach, and dots that can never light are not part of the panel
          if (d < hole || d > rim) continue
          map[gy * n + gx] = xs.length
          xs.push(x)
          ys.push(y)
        }
      }

      const total = xs.length
      dotX = Float32Array.from(xs)
      dotY = Float32Array.from(ys)
      outward = new Int32Array(total)
      value = new Float32Array(total)
      interior = new Uint8Array(total)
      edge = new Uint8Array(total)

      const at = (x: number, y: number): number => {
        const gx = Math.round(x / pitch + off)
        const gy = Math.round(y / pitch + off)
        return gx >= 0 && gx < n && gy >= 0 && gy < n ? map[gy * n + gx] : -1
      }
      const right = new Int32Array(total)
      const left = new Int32Array(total)
      const down = new Int32Array(total)
      const downRight = new Int32Array(total)
      const downLeft = new Int32Array(total)
      for (let i = 0; i < total; i++) {
        const x = dotX[i]
        const y = dotY[i]
        const d = Math.hypot(x, y) || 1
        const ux = (x / d) * pitch
        const uy = (y / d) * pitch
        outward[i] = at(x + ux, y + uy)
        // the ink's neighbours are the lattice's own, not the wave's radial
        // ones: ink flows between cells that are actually side by side, which
        // around a ring means sideways as much as outward
        right[i] = at(x + pitch, y)
        left[i] = at(x - pitch, y)
        down[i] = at(x, y + pitch)
        downRight[i] = at(x + pitch, y + pitch)
        downLeft[i] = at(x - pitch, y + pitch)
      }
      ink = {
        count: total,
        x: dotX,
        y: dotY,
        fill: new Float32Array(total),
        shade: new Uint8Array(total),
        right,
        left,
        down,
        downRight,
        downLeft,
      }
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
          // at rest, a slow breath. the same target on every spoke, so the
          // resting wave is a true circle: quantised onto the lattice it
          // reads as deliberate matrix geometry, where even a small fixed
          // per-spoke offset reads as a wobble in what should be a still ring
          target = 0.1 + 0.045 * Math.sin(now / 1300)
        }
        amp[s] += (target - amp[s]) * (target > amp[s] ? rise : fall)
      }

      buildWave(blobs, amp, {
        inner,
        span: inner * (OUTER / INNER - 1),
        pitch,
      })

      ctx.clearRect(0, 0, size, size)
      if (!ink) return
      const fill = ink.fill

      for (let i = 0; i < value.length; i++) {
        const v = blobs.at(dotX[i], dotY[i])
        value[i] = v
        interior[i] = v >= SURFACE ? 1 : 0
      }

      // the crest is read off the thresholded mask's radial neighbours, not
      // off a band of field values - the field falls off at a rate that
      // depends on how big the blobs currently are, so a rim cut by value is
      // two dots thick on a loud bar and gone on a quiet one. off either end
      // of the lattice counts as empty, so a wave that has run past the edge
      // still gets a line drawn along it.
      for (let i = 0; i < value.length; i++) {
        if (!interior[i]) {
          edge[i] = 0
          continue
        }
        const out = outward[i]
        edge[i] = out < 0 || !interior[out] ? 1 : 0
      }

      const mid = size / 2
      ctx.save()
      ctx.translate(mid, mid)

      // the wave is one liquid, painted as nested bodies rather than as
      // side-by-side shade chains. each layer is a superset of the one above,
      // so each is a continuous mass in its own right: the seams between
      // shades land INSIDE ink instead of being gaps between bead chains, and
      // a cell crossing a threshold changes which overlays cover it without
      // ever breaking the body underneath.

      // the skirt: everything the wave has breathed on, one dim mass that
      // saturates exactly at the surface, so the interior always sits on
      // solid ink. the curve is squared: near the surface it must be solid -
      // it is what the interior stands on - but out in the glow it should be
      // announcing the wave in small beads, not shadowing it in fat ones.
      for (let i = 0; i < fill.length; i++) {
        const t = ramp(WET_AT, SURFACE, value[i])
        fill[i] = t * t
      }
      paintInk(ctx, ink, inkGeo, [palette[WAVE_GLOW]])

      // the interior: the mass proper. gated on the mask with a floor rather
      // than ramped from the surface value, so the body reaches all the way
      // out under the crest riding its edge instead of receding a cell
      // inside it and leaving the boundary dots on bare skirt.
      for (let i = 0; i < fill.length; i++)
        fill[i] = interior[i] ? 0.5 + 0.5 * ramp(SURFACE, FULL_AT, value[i]) : 0
      paintInk(ctx, ink, inkGeo, [palette[WAVE_BODY]])

      // the crest rides on top: a one-dot bright line marking the surface.
      // struck as plain dots, not fused - a quantised circle crosses the
      // lattice as a staircase, and beads joined along a staircase read as a
      // worm weaving round the ring, not as a surface. the liquid is the mass
      // underneath; the line riding it is the honest matrix. a floor keeps it
      // legible - a boundary cell's field value is pinned near the surface by
      // definition, so filled straight off the field it would be pinpricks.
      for (let i = 0; i < fill.length; i++)
        fill[i] = edge[i] ? 0.55 + 0.45 * ramp(SURFACE, FULL_AT, value[i]) : 0
      paintInk(ctx, ink, lineGeo, [palette[WAVE_CREST]])

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
