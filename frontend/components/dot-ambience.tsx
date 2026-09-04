"use client"

import { useEffect, useRef } from "react"
import { BlobSet, shadeRamp, SURFACE, type Band } from "@/lib/dot-field"

interface DotAmbienceProps {
  // 0..1-ish, how loud the music is right now
  getLevel: () => number
  playing: boolean
}

// the panel's own weather: a handful of very large, very slow metaballs
// drifting behind everything else, struck on a coarse dot grid.
//
// it is the same field the cat and the wave are made of, read the other way -
// as a soft ramp rather than as a silhouette - and at an amplitude two ranks
// under the interface, so it never competes with anything you are trying to
// read. it exists because a lava lamp is the right ambience for this and
// because it ties the panel to the two things drawn on top of it: they all
// breathe on the same beat and lean towards the same cursor.
// twice the pitch of the panel's own unlit-dot field, and offset onto it, so
// every dot this draws lands exactly on one of the panel's. an unrelated pitch
// here reads as a second screen door laid over the first; on the same lattice
// it reads as the panel itself lighting up, which is the only excuse a layer
// like this has for being there at all.
const PITCH = 12
const PANEL_DOT = 2
// the blobs live in grid cells, so a radius of eight and a half is a hundred
// pixels of slow-moving weather
const R_BASE = 8.5

// four is enough to never repeat and few enough to be free. the periods are
// mutually irrational-ish on purpose - on round multiples they would all cross
// the middle together every few seconds and read as a pulse.
const DRIFT: Array<[number, number, number, number]> = [
  [0.19, 0.31, 0.24, 0.13],
  [0.13, 0.21, 0.62, 0.31],
  [0.27, 0.11, 0.41, 0.72],
  [0.09, 0.17, 0.77, 0.55],
]

// thirty frames a second. this is wallpaper and it moves slowly, but at twenty
// the blob edges creep across the grid in visible steps.
const MIN_FRAME_MS = 1000 / 30 - 4

const PALETTE_VARS = ["--dot-0", "--dot-1", "--dot-2", "--dot-3"]

// no silhouette here, only a mound: brightest where the blobs pile up on each
// other, fading out long before it reaches an edge. a rim would draw an
// outline round the wallpaper, which is the one thing wallpaper must not have.
const BANDS: Band[] = [
  [SURFACE * 2.4, 3],
  [SURFACE * 1.3, 2],
  [SURFACE * 0.4, 1],
]

export function DotAmbience({ getLevel, playing }: DotAmbienceProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const levelRef = useRef(getLevel)
  const playingRef = useRef(playing)
  levelRef.current = getLevel
  playingRef.current = playing

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const parent = canvas.parentElement
    if (!parent) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    let palette = PALETTE_VARS.map(() => "#000")
    const readPalette = () => {
      const style = getComputedStyle(document.documentElement)
      palette = PALETTE_VARS.map((v) => style.getPropertyValue(v).trim() || "#000")
    }
    readPalette()
    const themeWatch = new MutationObserver(readPalette)
    themeWatch.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] })

    let cols = 0
    let rows = 0
    let dpr = 0
    let field = new Float32Array(0)
    let grid = new Uint8Array(0)
    const blobs = new BlobSet()
    const counts = new Int32Array(PALETTE_VARS.length)
    let buckets: Float32Array[] = []

    const resize = (): boolean => {
      const nextDpr = Math.min(2, window.devicePixelRatio || 1)
      const w = parent.clientWidth
      const h = parent.clientHeight
      const nextCols = Math.ceil(w / PITCH) + 1
      const nextRows = Math.ceil(h / PITCH) + 1
      if (nextCols === cols && nextRows === rows && nextDpr === dpr) return false
      cols = nextCols
      rows = nextRows
      dpr = nextDpr
      canvas.width = w * dpr
      canvas.height = h * dpr
      canvas.style.width = `${w}px`
      canvas.style.height = `${h}px`
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      field = new Float32Array(cols * rows)
      grid = new Uint8Array(cols * rows)
      buckets = PALETTE_VARS.map(() => new Float32Array(cols * rows * 2))
      return true
    }
    resize()
    const sizeWatch = new ResizeObserver(resize)
    sizeWatch.observe(parent)

    // the cursor, in grid cells, so one blob can lean towards it
    let pullX = -1
    let pullY = -1
    const track = (e: PointerEvent) => {
      const box = parent.getBoundingClientRect()
      pullX = (e.clientX - box.left) / PITCH
      pullY = (e.clientY - box.top) / PITCH
    }
    window.addEventListener("pointermove", track, { passive: true })

    const dotRadius = 1.6
    let raf = 0
    let last = performance.now()
    let swell = 0

    const draw = (t: number, dt: number) => {
      // the music widens every blob at once, slowly. a fast follow here would
      // be a strobe behind the text, which is the opposite of ambience.
      const level = playingRef.current ? Math.min(1, levelRef.current() * 4) : 0
      swell += (level - swell) * (1 - Math.exp(-dt / 0.8))

      blobs.reset()
      const r = R_BASE * (1 + swell * 0.22)
      for (let i = 0; i < DRIFT.length; i++) {
        const [fx, fy, px, py] = DRIFT[i]
        blobs.add(
          cols * (0.5 + 0.42 * Math.sin(t * fx + px * 7)),
          rows * (0.5 + 0.46 * Math.sin(t * fy + py * 7)),
          r,
        )
      }
      // one more that hangs about wherever the cursor is, so the wallpaper
      // knows you are there too
      if (pullX >= 0) blobs.add(pullX, pullY, r * 0.72)

      blobs.scatter(field, cols, rows)
      shadeRamp(field, grid, BANDS)

      ctx.clearRect(0, 0, cols * PITCH, rows * PITCH)
      counts.fill(0)
      for (let i = 0; i < grid.length; i++) {
        const v = grid[i]
        if (v === 0) continue
        const at = counts[v]
        buckets[v][at] = (i % cols) * PITCH + PANEL_DOT
        buckets[v][at + 1] = Math.floor(i / cols) * PITCH + PANEL_DOT
        counts[v] = at + 2
      }
      for (let v = 1; v < buckets.length; v++) {
        const filled = counts[v]
        if (filled === 0) continue
        ctx.fillStyle = palette[v]
        ctx.beginPath()
        for (let i = 0; i < filled; i += 2) {
          ctx.moveTo(buckets[v][i] + dotRadius, buckets[v][i + 1])
          ctx.arc(buckets[v][i], buckets[v][i + 1], dotRadius, 0, Math.PI * 2)
        }
        ctx.fill()
      }
    }

    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      draw(0, 1)
      return () => {
        themeWatch.disconnect()
        sizeWatch.disconnect()
        window.removeEventListener("pointermove", track)
      }
    }

    const tick = (now: number) => {
      raf = requestAnimationFrame(tick)
      const since = now - last
      if (since < MIN_FRAME_MS) return
      last = now
      draw(now / 1000, Math.min(0.2, since / 1000))
    }
    raf = requestAnimationFrame(tick)

    return () => {
      cancelAnimationFrame(raf)
      themeWatch.disconnect()
      sizeWatch.disconnect()
      window.removeEventListener("pointermove", track)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      aria-hidden
      className="pointer-events-none absolute inset-0 z-0"
      style={{ opacity: 0.35 }}
    />
  )
}

export default DotAmbience
