"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { DotGlyph } from "@/components/dot-glyph"
import { DotSlider } from "@/components/dot-slider"

interface PomodoroTimerProps {
  onRunningChange: (running: boolean) => void
}

const SEGMENTS = 28

const formatTime = (seconds: number) => {
  const total = Math.max(0, Math.ceil(seconds))
  const mins = Math.floor(total / 60)
  const secs = total % 60
  return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`
}

export function PomodoroTimer({ onRunningChange }: PomodoroTimerProps) {
  const [workDuration, setWorkDuration] = useState(25)
  const [breakDuration, setBreakDuration] = useState(5)
  const [isBreak, setIsBreak] = useState(false)
  const [isRunning, setIsRunning] = useState(false)
  const [timeLeft, setTimeLeft] = useState(25 * 60)

  // the phase ends at a wall-clock instant, not after N ticks. counting ticks
  // drifts, and drifts more the longer the tab is backgrounded.
  const deadlineRef = useRef(0)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  // the interval closes over the phase; a ref is the only copy it can trust
  const isBreakRef = useRef(isBreak)
  isBreakRef.current = isBreak

  const total = (isBreak ? breakDuration : workDuration) * 60

  useEffect(() => {
    const audio = new Audio("/timer-end.mp3")
    audioRef.current = audio
    return () => {
      audio.pause()
      audio.removeAttribute("src")
      audio.load()
      if (audioRef.current === audio) audioRef.current = null
    }
  }, [])

  useEffect(() => {
    onRunningChange(isRunning)
  }, [isRunning, onRunningChange])

  useEffect(() => {
    if (!isRunning) return
    const id = setInterval(() => {
      const left = (deadlineRef.current - Date.now()) / 1000
      if (left <= 0) {
        setTimeLeft(0)
        setIsRunning(false)
        void audioRef.current?.play().catch(() => {})
        const wasBreak = isBreakRef.current
        setIsBreak(!wasBreak)
        setTimeLeft((wasBreak ? workDuration : breakDuration) * 60)
        return
      }
      setTimeLeft(left)
    }, 200)
    return () => clearInterval(id)
  }, [isRunning, workDuration, breakDuration])

  const toggle = useCallback(() => {
    setIsRunning((running) => {
      if (running) return false
      deadlineRef.current = Date.now() + timeLeft * 1000
      return true
    })
  }, [timeLeft])

  const reset = useCallback(() => {
    setIsRunning(false)
    setIsBreak(false)
    setTimeLeft(workDuration * 60)
  }, [workDuration])

  const setPhaseDuration = (minutes: number, forBreak: boolean) => {
    if (forBreak) setBreakDuration(minutes)
    else setWorkDuration(minutes)
    if (!isRunning && isBreak === forBreak) setTimeLeft(minutes * 60)
  }

  const elapsed = total > 0 ? 1 - timeLeft / total : 0
  const lit = Math.round(Math.min(1, Math.max(0, elapsed)) * SEGMENTS)

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-baseline justify-between">
        <span className="label">{isBreak ? "Break" : "Focus"}</span>
        <span className="label">{isRunning ? "running" : "held"}</span>
      </div>

      <div className="flex items-center justify-between gap-4">
        <span
          className="tabular-nums leading-none"
          style={{
            fontSize: "2.6rem",
            letterSpacing: "0.02em",
            color: isRunning ? "var(--accent)" : "var(--text)",
          }}
        >
          {formatTime(timeLeft)}
        </span>

        <div className="flex gap-2">
          <button
            type="button"
            onClick={toggle}
            className="key h-9 w-9"
            aria-label={isRunning ? "Pause timer" : "Start timer"}
          >
            <DotGlyph name={isRunning ? "pause" : "play"} dot={2} />
          </button>
          <button type="button" onClick={reset} className="key h-9 w-9" aria-label="Reset timer">
            <DotGlyph name="rewind" dot={2} />
          </button>
        </div>
      </div>

      <div className="flex items-center justify-between" aria-hidden>
        {Array.from({ length: SEGMENTS }, (_, i) => (
          <span
            key={i}
            className="rounded-full"
            style={{
              width: 4,
              height: 4,
              background: i < lit ? "var(--accent)" : "var(--dot-1)",
            }}
          />
        ))}
      </div>

      <div className="grid grid-cols-2 gap-x-5 gap-y-3">
        <DotSlider
          label="Work"
          readout={`${workDuration}m`}
          value={workDuration}
          min={1}
          max={60}
          segments={10}
          disabled={isRunning}
          onChange={(v) => setPhaseDuration(v, false)}
        />
        <DotSlider
          label="Rest"
          readout={`${breakDuration}m`}
          value={breakDuration}
          min={1}
          max={30}
          segments={10}
          disabled={isRunning}
          onChange={(v) => setPhaseDuration(v, true)}
        />
      </div>
    </div>
  )
}

export default PomodoroTimer
