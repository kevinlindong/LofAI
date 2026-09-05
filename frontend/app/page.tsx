"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { MetaballField } from "@/components/metaball-field"
import { MusicControls } from "@/components/music-controls"
import { Pet, type PetEvent, type PetSignal } from "@/components/pet"
import { PomodoroTimer } from "@/components/pomodoro-timer"
import { ThemeToggle } from "@/components/theme-toggle"
import { TodoList } from "@/components/todo-list"
import {
  DEFAULT_LISTENER_CONTROLS,
  MrtStream,
  type ListenerControls,
  type StreamState,
} from "@/lib/mrt-stream"

const IDLE_STATE: StreamState = {
  status: "idle",
  queuePosition: 0,
  listeners: 0,
  capacity: 0,
  message: null,
  bufferProgress: 0,
  variationPending: false,
}

// the header carries the connection state; the transport carries the detail
const SHORT_STATUS: Record<string, string> = {
  idle: "standby",
  connecting: "linking",
  loading: "loading",
  queued: "queued",
  buffering: "buffering",
  live: "on air",
  paused: "held",
  error: "fault",
}

function statusLabel(state: StreamState, wantsAudio: boolean): string {
  if (state.variationPending) return "finding a new take"
  switch (state.status) {
    case "loading":
      return state.message ?? "warming up the model"
    case "connecting":
      return state.message ?? "connecting"
    case "queued":
      return state.queuePosition > 0
        ? `waiting for a slot · #${state.queuePosition}`
        : "waiting for a slot"
    case "buffering":
      // this machine renders a touch slower than real time, so the stream
      // banks a reservoir before it plays and tops it up when it runs out
      return `catching up · ${Math.round(state.bufferProgress * 100)}%`
    case "live":
      return "live"
    case "paused":
      return "held"
    case "error":
      return state.message ?? "something broke"
    default:
      return wantsAudio ? "starting" : "ready"
  }
}

export default function LofiGenerator() {
  const [controls, setControls] = useState<ListenerControls>(() => ({
    ...DEFAULT_LISTENER_CONTROLS,
  }))
  const [volume, setVolume] = useState(100)
  const [wantsAudio, setWantsAudio] = useState(false)
  const [streamState, setStreamState] = useState<StreamState>(IDLE_STATE)
  const [petSignal, setPetSignal] = useState<PetSignal | null>(null)
  const [focusMode, setFocusMode] = useState(false)
  const streamRef = useRef<MrtStream | null>(null)

  const isLive = streamState.status === "live"

  useEffect(() => {
    const stream = new MrtStream(setStreamState)
    streamRef.current = stream
    return () => {
      stream.destroy()
      streamRef.current = null
    }
  }, [])

  // The stream retains this state before connecting, then sends one small
  // station/drum update whenever the listener changes it.
  useEffect(() => {
    streamRef.current?.setControls(controls)
  }, [controls])

  useEffect(() => {
    streamRef.current?.setVolume(volume / 100)
  }, [volume])

  // A generation failure stops the stream internally. Reflect that transport
  // state in React so the next click is a genuine retry instead of another
  // pause request against an already suspended backend session.
  useEffect(() => {
    if (streamState.status === "error" && wantsAudio) setWantsAudio(false)
  }, [streamState.status, wantsAudio])

  const togglePlayback = useCallback(async () => {
    const stream = streamRef.current
    if (!stream) return

    if (wantsAudio) {
      stream.pause()
      setWantsAudio(false)
      return
    }

    setWantsAudio(true)
    await stream.start(controls)
  }, [wantsAudio, controls])

  const requestVariation = useCallback(() => {
    streamRef.current?.newVariation()
  }, [])

  const getLevel = useCallback(() => streamRef.current?.level() ?? 0, [])
  const getSpectrum = useCallback(
    (out: Uint8Array) => streamRef.current?.spectrum(out) ?? 0,
    [],
  )

  const handlePetEvent = useCallback((kind: PetEvent) => {
    setPetSignal({ kind, at: Date.now() })
  }, [])

  const label = useMemo(() => statusLabel(streamState, wantsAudio), [streamState, wantsAudio])

  return (
    <main className="site-shell">
      <MetaballField getLevel={getLevel} playing={isLive} />

      <div className="app-frame">
        <header className="app-header">
          <div className="brand-lockup min-w-0">
            <div className="brand-flow" aria-hidden>
              <span />
              <span />
              <span />
            </div>
            <div className="min-w-0">
              <h1 className="brand-name dot-type">LOFAI</h1>
              <p className="brand-tagline hidden sm:block">endless lofi, shaped while it plays</p>
            </div>
          </div>

          <div className="flex items-center gap-2 sm:gap-3 shrink-0">
            <span className="status-pill">
              <span
                className={`status-orb ${isLive ? "is-live" : ""}`}
                style={{
                  background: isLive
                    ? "var(--good)"
                    : streamState.status === "error"
                      ? "var(--bad)"
                      : "var(--text-faint)",
                }}
              />
              <span className="hidden sm:inline">
                {SHORT_STATUS[streamState.status] ?? "standby"}
              </span>
            </span>
            <ThemeToggle />
          </div>
        </header>

        <div className="workspace-grid">
          <section className="surface-card music-card">
            <div className="card-intro">
              <div>
                <p className="eyebrow">Generative radio</p>
                <h2>Find your flow.</h2>
              </div>
              <p className="card-note hidden sm:block">A live soundtrack that changes with you.</p>
            </div>
            <MusicControls
              isPlaying={wantsAudio}
              togglePlayback={togglePlayback}
              requestVariation={requestVariation}
              variationPending={streamState.variationPending}
              controls={controls}
              setControls={setControls}
              volume={volume}
              setVolume={setVolume}
              statusLabel={label}
              isLive={isLive}
              getSpectrum={getSpectrum}
            />
          </section>

          <div className="side-stack">
            <section className="surface-card companion-card">
              <Pet
                signal={petSignal}
                focus={focusMode}
                playing={isLive}
                getLevel={getLevel}
              />
            </section>

            <section className="surface-card tasks-card">
              <TodoList onEvent={handlePetEvent} />
            </section>

            <section className="surface-card timer-card">
              <PomodoroTimer onRunningChange={setFocusMode} />
            </section>
          </div>
        </div>
      </div>
    </main>
  )
}
