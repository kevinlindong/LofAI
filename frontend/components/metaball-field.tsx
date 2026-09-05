"use client"

import { useEffect, useRef } from "react"

interface MetaballFieldProps {
  getLevel: () => number
  playing: boolean
}

interface Track {
  x: number
  y: number
  moveX: number
  moveY: number
  speedX: number
  speedY: number
  phaseX: number
  phaseY: number
  radius: number
  tone: number
}

// These paths overlap on purpose. A collection of circles simply drifting in
// parallel still reads as decoration; crossing paths let the scalar field
// create the necks, stretches and separations that make metaballs feel fluid.
const TRACKS: Track[] = [
  {
    x: 0.12,
    y: 0.2,
    moveX: 0.23,
    moveY: 0.2,
    speedX: 0.13,
    speedY: 0.17,
    phaseX: 0.2,
    phaseY: 1.7,
    radius: 0.12,
    tone: 0,
  },
  {
    x: 0.42,
    y: 0.24,
    moveX: 0.26,
    moveY: 0.17,
    speedX: 0.1,
    speedY: 0.14,
    phaseX: 2.2,
    phaseY: 0.4,
    radius: 0.09,
    tone: 0,
  },
  {
    x: 0.77,
    y: 0.18,
    moveX: 0.18,
    moveY: 0.25,
    speedX: 0.12,
    speedY: 0.09,
    phaseX: 4.1,
    phaseY: 2.3,
    radius: 0.14,
    tone: 1,
  },
  {
    x: 0.24,
    y: 0.7,
    moveX: 0.2,
    moveY: 0.2,
    speedX: 0.08,
    speedY: 0.13,
    phaseX: 5.2,
    phaseY: 1.1,
    radius: 0.105,
    tone: 2,
  },
  {
    x: 0.58,
    y: 0.72,
    moveX: 0.29,
    moveY: 0.16,
    speedX: 0.11,
    speedY: 0.16,
    phaseX: 0.9,
    phaseY: 3.8,
    radius: 0.12,
    tone: 1,
  },
  {
    x: 0.86,
    y: 0.68,
    moveX: 0.21,
    moveY: 0.22,
    speedX: 0.07,
    speedY: 0.115,
    phaseX: 3.4,
    phaseY: 5.1,
    radius: 0.085,
    tone: 2,
  },
]

const FRAME_MS = 1000 / 30
const SAMPLE_SIZE = 5

const smoothstep = (edge0: number, edge1: number, value: number) => {
  const t = Math.min(1, Math.max(0, (value - edge0) / (edge1 - edge0)))
  return t * t * (3 - 2 * t)
}

const readRgb = (style: CSSStyleDeclaration, property: string, fallback: number[]) => {
  const values = style
    .getPropertyValue(property)
    .trim()
    .split(/\s+/)
    .map(Number)
  return values.length === 3 && values.every(Number.isFinite) ? values : fallback
}

/**
 * A low-resolution scalar field, enlarged with interpolation. Each moving
 * circle contributes r²/d² to the field; wherever their combined influence
 * crosses the surface, separate dots become one continuous shape.
 */
export function MetaballField({ getLevel, playing }: MetaballFieldProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const levelRef = useRef(getLevel)
  const playingRef = useRef(playing)
  levelRef.current = getLevel
  playingRef.current = playing

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d", { alpha: true })
    if (!ctx) return

    let width = 0
    let height = 0
    let image = ctx.createImageData(1, 1)
    let palette: number[][] = []
    let fieldOpacity = 0.35
    let redrawStill: (() => void) | null = null

    const readPalette = () => {
      const style = getComputedStyle(document.documentElement)
      palette = [
        readRgb(style, "--blob-a", [244, 140, 91]),
        readRgb(style, "--blob-b", [80, 177, 154]),
        readRgb(style, "--blob-c", [223, 177, 91]),
      ]
      fieldOpacity = Number(style.getPropertyValue("--blob-opacity")) || 0.35
      redrawStill?.()
    }
    readPalette()

    const resize = () => {
      const rect = canvas.getBoundingClientRect()
      const nextWidth = Math.max(1, Math.ceil(rect.width / SAMPLE_SIZE))
      const nextHeight = Math.max(1, Math.ceil(rect.height / SAMPLE_SIZE))
      if (nextWidth === width && nextHeight === height) return
      width = nextWidth
      height = nextHeight
      canvas.width = width
      canvas.height = height
      image = ctx.createImageData(width, height)
      redrawStill?.()
    }
    resize()

    const themeWatch = new MutationObserver(readPalette)
    themeWatch.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    })
    const sizeWatch = new ResizeObserver(resize)
    sizeWatch.observe(canvas)

    const pointer = { x: 0.5, y: 0.5, targetX: 0.5, targetY: 0.5, at: -Infinity }
    const trackPointer = (event: PointerEvent) => {
      pointer.targetX = event.clientX / Math.max(1, window.innerWidth)
      pointer.targetY = event.clientY / Math.max(1, window.innerHeight)
      pointer.at = performance.now()
    }
    window.addEventListener("pointermove", trackPointer, { passive: true })

    // x, y, radius and tone for the six ambient balls plus the cursor ball.
    const balls = new Float32Array((TRACKS.length + 1) * 4)
    let swell = 0

    const draw = (now: number, dt: number) => {
      if (width <= 1 || height <= 1) return

      const level = playingRef.current ? Math.min(1, levelRef.current() * 4) : 0
      swell += (level - swell) * (1 - Math.exp(-dt / 0.65))
      const aspect = width / height
      const radiusScale = Math.max(0.72, Math.min(1.18, Math.sqrt(aspect)))
      const t = now / 1000

      TRACKS.forEach((track, index) => {
        const at = index * 4
        balls[at] = track.x + Math.sin(t * track.speedX + track.phaseX) * track.moveX
        balls[at + 1] = track.y + Math.sin(t * track.speedY + track.phaseY) * track.moveY
        balls[at + 2] = track.radius * radiusScale * (1 + swell * 0.16)
        balls[at + 3] = track.tone
      })

      const follow = 1 - Math.exp(-dt / 0.22)
      pointer.x += (pointer.targetX - pointer.x) * follow
      pointer.y += (pointer.targetY - pointer.y) * follow
      const pointerLife = Math.max(0, 1 - (now - pointer.at) / 4200)
      const pointerAt = TRACKS.length * 4
      balls[pointerAt] = pointer.x
      balls[pointerAt + 1] = pointer.y
      balls[pointerAt + 2] = 0.064 * radiusScale * pointerLife
      balls[pointerAt + 3] = 1

      const pixels = image.data
      const ballCount = TRACKS.length + (pointerLife > 0 ? 1 : 0)
      let pixel = 0

      for (let y = 0; y < height; y++) {
        const ny = (y + 0.5) / height
        for (let x = 0; x < width; x++) {
          const nx = (x + 0.5) / width
          let field = 0
          let red = 0
          let green = 0
          let blue = 0

          for (let b = 0; b < ballCount; b++) {
            const at = b * 4
            // Correct x for the viewport aspect so a ball stays circular.
            const dx = (nx - balls[at]) * aspect
            const dy = ny - balls[at + 1]
            const radius = balls[at + 2]
            const influence = (radius * radius) / (dx * dx + dy * dy + 0.000012)
            if (influence < 0.065) continue
            const colour = palette[balls[at + 3]]
            field += influence
            red += colour[0] * influence
            green += colour[1] * influence
            blue += colour[2] * influence
          }

          // A narrow threshold creates the characteristic liquid bridge. A
          // quieter outer band keeps separations from popping on and off.
          const body = smoothstep(0.72, 1.06, field)
          const haze = smoothstep(0.31, 0.78, field)
          const alpha = Math.min(1, body * 0.9 + haze * 0.1) * fieldOpacity
          const divisor = field || 1
          const glow = smoothstep(1.1, 3.1, field) * 12

          pixels[pixel] = Math.min(255, red / divisor + glow)
          pixels[pixel + 1] = Math.min(255, green / divisor + glow)
          pixels[pixel + 2] = Math.min(255, blue / divisor + glow)
          pixels[pixel + 3] = Math.round(alpha * 255)
          pixel += 4
        }
      }

      ctx.putImageData(image, 0, 0)
    }

    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      redrawStill = () => draw(0, 1)
      redrawStill()
      return () => {
        themeWatch.disconnect()
        sizeWatch.disconnect()
        window.removeEventListener("pointermove", trackPointer)
      }
    }

    let raf = 0
    let last = performance.now()
    const tick = (now: number) => {
      raf = requestAnimationFrame(tick)
      const elapsed = now - last
      if (elapsed < FRAME_MS) return
      last = now
      draw(now, Math.min(0.15, elapsed / 1000))
    }
    raf = requestAnimationFrame(tick)

    return () => {
      cancelAnimationFrame(raf)
      themeWatch.disconnect()
      sizeWatch.disconnect()
      window.removeEventListener("pointermove", trackPointer)
    }
  }, [])

  return (
    <div className="metaball-field" aria-hidden>
      <canvas ref={canvasRef} />
    </div>
  )
}

export default MetaballField
