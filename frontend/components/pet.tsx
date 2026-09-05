"use client"

import { useEffect, useRef, useState } from "react"
import {
  drawPet,
  IDLE_FRAME,
  PET_H,
  PET_W,
  type PetFrame,
  type PetMood,
} from "@/lib/pet-scene"

export type PetEvent = "add" | "complete" | "clear" | "undo"

// what the cat is reacting to. "pet" is not a PetEvent because nothing outside
// this component raises it - it is the cat noticing that you clicked on it.
type Reaction = PetEvent | "pet"

export interface PetSignal {
  kind: PetEvent
  // bumped on every signal so the same event twice in a row still lands
  at: number
}

interface PetProps {
  signal: PetSignal | null
  // the pomodoro is running, so the cat is working too
  focus: boolean
  playing: boolean
  getLevel: () => number
}

const PALETTE_VARS = [
  "--dot-0",
  "--dot-1",
  "--dot-2",
  "--dot-3",
  "--dot-4",
  "--dot-5",
  "--dot-6",
  "--dot-7",
]

// how long each reaction holds the cat's face
const HOLD_MS: Record<Reaction, number> = {
  add: 750,
  complete: 1900,
  clear: 3200,
  undo: 700,
  pet: 1400,
}

// every frame the display offers. thirty was the old setting, on the argument
// that a dot matrix reads the same at half the rate and the other half of the
// budget belongs to the music. that argument is wrong about the shape changes:
// the squash, the ear flick and the tail curl are continuous, and at thirty
// they stutter. building a frame costs a fortieth of a millisecond, so there
// is nothing to save.
const MIN_FRAME_MS = 0

// how the head follows the beat, as time constants in seconds: drops fast,
// comes back up slowly. seconds rather than per-frame fractions so the bob
// feels the same however often we draw. the attack is a shade slower than it
// wants to be on paper - the head moves in whole dots, so an attack quick
// enough to catch every transient just makes it flicker between two rows.
const BOB_ATTACK = 0.055
const BOB_RELEASE = 0.27

// how fast the eyes catch up with the cursor. slow enough to be a head turning
// rather than a cursor with whiskers, fast enough not to feel broken.
const GAZE_FOLLOW = 0.13

// the cursor has to travel about this far from the cat's eye to pull the look
// all the way over. roughly the width of the picture, so anywhere on the panel
// gets a look and anywhere across the page gets the full one.
const GAZE_SPAN_X = 240
const GAZE_SPAN_Y = 170

// the cat gives up on a cursor that has not moved and goes back to its own
// business, which for a cat that is meant to be working is the page
const ATTENTION_MS = 4000

// where the eyes sit in the picture, in dots, for working out what the cursor
// is off to the side of
const EYE_C = 16
const EYE_R = 12

// the caption is the other half of the animation. a tamagotchi tells you what
// it is doing in words as well as in pixels, and at forty dots across the
// words carry more of it than you would like to admit.
const CAPTIONS: Record<PetMood, string> = {
  sleep: "dozing",
  cheer: "delighted",
  happy: "pleased",
  purr: "purring",
  focus: "keeping watch",
  bop: "bopping",
  idle: "loafing",
}

export function Pet({ signal, focus, playing, getLevel }: PetProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const wrapRef = useRef<HTMLDivElement>(null)
  const [caption, setCaption] = useState("idling")

  // everything the animation loop reads lives in refs: the loop runs at frame
  // rate and must never be the reason react re-renders
  const reactionRef = useRef<{ kind: Reaction; start: number } | null>(null)
  const focusRef = useRef(focus)
  const playingRef = useRef(playing)
  const levelRef = useRef(getLevel)
  const lastPokeRef = useRef(Date.now())
  // the cursor, in client coordinates, and when it last actually moved
  const pointerRef = useRef<{ x: number; y: number; at: number } | null>(null)

  focusRef.current = focus
  playingRef.current = playing
  levelRef.current = getLevel

  useEffect(() => {
    if (!signal) return
    reactionRef.current = { kind: signal.kind, start: performance.now() }
    lastPokeRef.current = Date.now()
  }, [signal])

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

    // the palette lives in css variables, so a theme flip has to be watched for
    const themeWatch = new MutationObserver(() => {
      readPalette()
      redrawStill?.()
    })
    themeWatch.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    })

    let pitch = 0
    let dpr = 0
    // assigning canvas.width wipes the canvas, so only assign when something
    // actually changed - otherwise the observer's first callback erases the
    // frame that has just been drawn
    const resize = (): boolean => {
      const nextDpr = Math.min(2, window.devicePixelRatio || 1)
      // whole-pixel pitch only, and capped: past about six pixels the dots stop
      // reading as a matrix and start reading as circles, and the cat stops
      // being a small thing on a shelf. the floor is what the drawing needs to
      // still be a cat.
      const nextPitch = Math.min(6, Math.max(3, Math.floor(wrap.clientWidth / PET_W)))
      if (nextPitch === pitch && nextDpr === dpr) return false
      pitch = nextPitch
      dpr = nextDpr
      canvas.width = PET_W * pitch * dpr
      canvas.height = PET_H * pitch * dpr
      canvas.style.width = `${PET_W * pitch}px`
      canvas.style.height = `${PET_H * pitch}px`
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      return true
    }
    resize()
    const sizeWatch = new ResizeObserver(() => {
      if (resize()) redrawStill?.()
    })
    sizeWatch.observe(wrap)

    const paint = (frame: PetFrame) => {
      const grid = drawPet(frame)
      const r = Math.max(1, pitch * 0.4)
      ctx.clearRect(0, 0, PET_W * pitch, PET_H * pitch)
      // one path per colour: 1500 dots is nothing, 1500 state changes is not
      for (let v = 0; v < palette.length; v++) {
        ctx.fillStyle = palette[v]
        ctx.beginPath()
        for (let i = 0; i < grid.length; i++) {
          if (grid[i] !== v) continue
          const x = (i % PET_W) * pitch + pitch / 2
          const y = Math.floor(i / PET_W) * pitch + pitch / 2
          ctx.moveTo(x + r, y)
          ctx.arc(x, y, r, 0, Math.PI * 2)
        }
        ctx.fill()
      }
    }

    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      const resting: PetFrame = { ...IDLE_FRAME, phase: 1 }
      redrawStill = () => paint(resting)
      redrawStill()
      return () => {
        themeWatch.disconnect()
        sizeWatch.disconnect()
      }
    }

    let raf = 0
    let smoothed = 0
    let gazeX = 0
    let gazeY = 0
    let affection = 0
    let shownMood: PetMood | null = null
    let last = performance.now()
    let nextBlink = last + 2600
    let blinkUntil = 0

    const tick = (now: number) => {
      raf = requestAnimationFrame(tick)
      const since = now - last
      if (since < MIN_FRAME_MS) return
      const dt = Math.min(0.1, since / 1000)
      last = now

      // fast attack, slow release - the head drops on the beat and comes back
      // up between them, instead of vibrating at frame rate
      const level = Math.min(1, levelRef.current() * 4)
      smoothed +=
        (level - smoothed) *
        (1 - Math.exp(-dt / (level > smoothed ? BOB_ATTACK : BOB_RELEASE)))

      // ---- where the cat is looking ----
      const rect = canvas.getBoundingClientRect()
      const pointer = pointerRef.current
      const watching = pointer !== null && Date.now() - pointer.at < ATTENTION_MS
      let wantX: number
      let wantY: number
      let onCat = false

      if (watching && pointer) {
        // saturating, so the cursor two panels away and the cursor ten look the
        // same - past a certain point a head is simply turned as far as it goes
        const eyeX = rect.left + ((EYE_C + 0.5) / PET_W) * rect.width
        const eyeY = rect.top + ((EYE_R + 0.5) / PET_H) * rect.height
        wantX = Math.tanh((pointer.x - eyeX) / GAZE_SPAN_X)
        wantY = Math.tanh((pointer.y - eyeY) / GAZE_SPAN_Y)
        onCat =
          pointer.x >= rect.left &&
          pointer.x <= rect.right &&
          pointer.y >= rect.top &&
          pointer.y <= rect.bottom
      } else if (focusRef.current) {
        // eyes down and steady on whatever it is the two of you are doing
        wantX = -0.15
        wantY = 0.6
      } else {
        // nobody about: a slow, uneven wander, the two axes on different
        // periods so it never traces the same little circle twice
        wantX = Math.sin(now / 3100) * 0.4
        wantY = Math.sin(now / 4700) * 0.3 + 0.1
      }

      const follow = 1 - Math.exp(-dt / GAZE_FOLLOW)
      gazeX += (wantX - gazeX) * follow
      gazeY += (wantY - gazeY) * follow
      // warms quickly, cools slowly, so a cursor passing through does not
      // switch the cat on and off
      affection += ((onCat ? 1 : 0) - affection) * (1 - Math.exp(-dt / (onCat ? 0.2 : 0.7)))
      if (onCat) lastPokeRef.current = Date.now()

      if (now > nextBlink) {
        blinkUntil = now + 130
        nextBlink = now + 2600 + Math.random() * 4200
      }

      const reaction = reactionRef.current
      let mood: PetMood = "idle"
      let sparkle = 0
      let hop = 0
      let twitch = 0
      let pat = 0

      if (reaction) {
        const t = (now - reaction.start) / HOLD_MS[reaction.kind]
        // reactions are shaped in seconds, not in fractions of their hold: a
        // twitch is a twitch whether the pose it interrupts lasts half a
        // second or three
        const secs = (now - reaction.start) / 1000
        if (t >= 1) {
          reactionRef.current = null
        } else if (reaction.kind === "add" || reaction.kind === "undo") {
          // a task arriving is worth noticing but not celebrating: an ear goes
          // back and comes down again. a decaying wobble rather than a square
          // wave - an ear that snaps between two positions three times reads
          // as a fault in the panel.
          twitch = Math.max(0, Math.exp(-secs * 4) * Math.cos(secs * 13))
        } else if (reaction.kind === "pet") {
          // being fussed. this used to be a hop, which is what a cat does when
          // you drop something, not when you put your hand on it. a pat
          // presses the loaf down into the ground and it springs most of the
          // way back - the cosine going briefly negative is that rebound.
          mood = "purr"
          pat = Math.max(-0.35, Math.exp(-secs * 4.5) * Math.cos(secs * 7.5))
        } else {
          mood = reaction.kind === "clear" ? "cheer" : "happy"
          sparkle = 1 - t
          // one hop for a task, three for clearing the board
          const hops = reaction.kind === "clear" ? 3 : 1
          hop = Math.max(0, Math.sin(t * Math.PI * hops))
        }
      }

      // the state it settles into when nothing has just happened to it. the
      // order is the priority: a hand on the cat beats the pomodoro, the
      // pomodoro beats the music, and going to sleep needs all three quiet.
      if (mood === "idle") {
        const idleFor = Date.now() - lastPokeRef.current
        if (affection > 0.55) mood = "purr"
        else if (focusRef.current) mood = "focus"
        else if (playingRef.current) mood = "bop"
        else if (idleFor > 45_000) mood = "sleep"
      }

      if (mood !== shownMood) {
        shownMood = mood
        setCaption(CAPTIONS[mood])
      }

      paint({
        mood,
        blink: now < blinkUntil,
        bob: playingRef.current ? Math.min(1, smoothed * 1.6) : 0,
        hop,
        twitch,
        pat,
        phase: now / 1000,
        notes: playingRef.current,
        sparkle,
        gazeX,
        gazeY,
        affection,
      })
    }
    raf = requestAnimationFrame(tick)

    return () => {
      cancelAnimationFrame(raf)
      themeWatch.disconnect()
      sizeWatch.disconnect()
    }
  }, [])

  // any interaction anywhere counts as company, and wakes the cat
  useEffect(() => {
    const poke = () => {
      lastPokeRef.current = Date.now()
    }
    // a click on the cat itself, as opposed to a click anywhere on the page
    const press = (e: PointerEvent) => {
      poke()
      const box = canvasRef.current?.getBoundingClientRect()
      if (!box) return
      if (
        e.clientX >= box.left &&
        e.clientX <= box.right &&
        e.clientY >= box.top &&
        e.clientY <= box.bottom
      ) {
        reactionRef.current = { kind: "pet", start: performance.now() }
      }
    }
    const track = (e: PointerEvent) => {
      pointerRef.current = { x: e.clientX, y: e.clientY, at: Date.now() }
      poke()
    }
    const forget = () => {
      pointerRef.current = null
    }
    window.addEventListener("pointermove", track, { passive: true })
    window.addEventListener("pointerdown", press)
    window.addEventListener("keydown", poke)
    // the cursor leaving the window is not the cursor sitting still somewhere
    document.addEventListener("pointerleave", forget)
    return () => {
      window.removeEventListener("pointermove", track)
      window.removeEventListener("pointerdown", press)
      window.removeEventListener("keydown", poke)
      document.removeEventListener("pointerleave", forget)
    }
  }, [])

  return (
    <div className="flex w-full flex-col gap-3">
      <div className="flex items-baseline justify-between">
        <span className="label">Companion</span>
        <span className="label">{caption}</span>
      </div>
      <div className="pet-stage mx-auto w-full max-w-[18rem] p-2">
        {/* the pitch is measured off this, so it carries no padding of its own */}
        <div ref={wrapRef} className="flex justify-center">
          <canvas ref={canvasRef} aria-hidden />
        </div>
      </div>
    </div>
  )
}

export default Pet
