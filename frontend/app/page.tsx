"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { DotAmbience } from "@/components/dot-ambience"
import { MusicControls } from "@/components/music-controls"
import { Pet, type PetEvent, type PetSignal } from "@/components/pet"
import { PomodoroTimer } from "@/components/pomodoro-timer"
import { ThemeToggle } from "@/components/theme-toggle"
import { TodoList } from "@/components/todo-list"
import { MrtStream, type StreamState } from "@/lib/mrt-stream"

const IDLE_STATE: StreamState = {
  status: "idle",
  queuePosition: 0,
  listeners: 0,
  capacity: 0,
  message: null,
  bufferProgress: 0,
}

const getMoodText = (value: number) => (value < 25 ? "somber" : value < 75 ? "neutral" : "lively")

const getInstrumentType = (value: number) =>
  value < 25 ? "piano" : value < 75 ? "guitar" : "brass"

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
  const [mood, setMood] = useState(50)
  const [instrument, setInstrument] = useState(50)
  const [volume, setVolume] = useState(100)
  const [wantsAudio, setWantsAudio] = useState(false)
  const [streamState, setStreamState] = useState<StreamState>(IDLE_STATE)
  const [petSignal, setPetSignal] = useState<PetSignal | null>(null)
  const [focusMode, setFocusMode] = useState(false)
  const streamRef = useRef<MrtStream | null>(null)

  const moodText = getMoodText(mood)
  const instrumentText = getInstrumentType(instrument)
  const isLive = streamState.status === "live"

  useEffect(() => {
    const stream = new MrtStream(setStreamState)
    streamRef.current = stream
    return () => {
      stream.destroy()
      streamRef.current = null
    }
  }, [])

  // steer the running stream the moment the slider crosses into a new setting
  useEffect(() => {
    streamRef.current?.setStyle(moodText, instrumentText)
  }, [moodText, instrumentText])

  useEffect(() => {
    streamRef.current?.setVolume(volume / 100)
  }, [volume])

  const togglePlayback = useCallback(async () => {
    const stream = streamRef.current
    if (!stream) return

    if (wantsAudio) {
      stream.pause()
      setWantsAudio(false)
      return
    }

    setWantsAudio(true)
    await stream.start(getMoodText(mood), getInstrumentType(instrument))
  }, [wantsAudio, mood, instrument])

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
    <main className="min-h-screen p-3 sm:p-6 flex items-center justify-center">
      <div className="panel w-full max-w-6xl overflow-hidden">
        <DotAmbience getLevel={getLevel} playing={isLive} />

        <header className="relative z-10 flex items-center justify-between gap-4 border-b px-5 py-3">
          <div className="flex items-baseline gap-4 min-w-0">
            {/* uppercase on purpose: a lowercase l is indistinguishable from a 1 here */}
            <h1 style={{ fontSize: "1.35rem", letterSpacing: "0.2em" }}>LOFAI</h1>
            <p className="label hidden sm:block truncate">endless lofi, steered as it plays</p>
          </div>

          <div className="flex items-center gap-4 shrink-0">
            <span className="flex items-center gap-2">
              <span
                className={`rounded-full ${isLive ? "animate-dot-blink" : ""}`}
                style={{
                  width: 7,
                  height: 7,
                  background: isLive
                    ? "var(--good)"
                    : streamState.status === "error"
                      ? "var(--bad)"
                      : "var(--dot-1)",
                }}
              />
              <span className="label hidden sm:inline">
                {SHORT_STATUS[streamState.status] ?? "standby"}
              </span>
            </span>
            <ThemeToggle />
          </div>
        </header>

        <div className="relative z-10 grid lg:grid-cols-[1.1fr_1fr]">
          <section className="border-b lg:border-b-0 lg:border-r">
            <MusicControls
              isPlaying={wantsAudio}
              togglePlayback={togglePlayback}
              mood={mood}
              setMood={setMood}
              instrument={instrument}
              setInstrument={setInstrument}
              volume={volume}
              setVolume={setVolume}
              statusLabel={label}
              moodText={moodText}
              instrumentText={instrumentText}
              isLive={isLive}
              getSpectrum={getSpectrum}
            />
          </section>

          <section className="flex flex-col">
            <div className="border-b px-5 py-4">
              <Pet
                signal={petSignal}
                focus={focusMode}
                playing={isLive}
                getLevel={getLevel}
              />
            </div>

            {/* the tasks take whatever the cat has left over, which is what
                keeps the two columns the same height */}
            <div className="flex min-h-[16rem] flex-1 flex-col border-b px-5 py-4">
              <TodoList onEvent={handlePetEvent} />
            </div>

            <div className="px-5 py-4">
              <PomodoroTimer onRunningChange={setFocusMode} />
            </div>
          </section>
        </div>
      </div>
    </main>
  )
}
