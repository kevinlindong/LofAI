"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { AsciiAmbience } from "@/components/ascii-ambience"
import { MusicControls } from "@/components/music-controls"
import { Pet, type PetEvent, type PetSignal } from "@/components/pet"
import { PomodoroTimer } from "@/components/pomodoro-timer"
import { PageMenu } from "@/components/page-menu"
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
      <AsciiAmbience />

      <div className="app-frame">
        <PageMenu />

        <div className="workspace-grid">
          <section id="radio" className="surface-card music-card" tabIndex={-1}>
            <div className="card-intro">
              <div>
                <p className="eyebrow">Generative radio</p>
                <h1>Find your flow.</h1>
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

            <section id="tasks" className="surface-card tasks-card" tabIndex={-1}>
              <TodoList onEvent={handlePetEvent} />
            </section>

            <section id="focus-timer" className="surface-card timer-card" tabIndex={-1}>
              <PomodoroTimer onRunningChange={setFocusMode} />
            </section>
          </div>
        </div>
      </div>
    </main>
  )
}
