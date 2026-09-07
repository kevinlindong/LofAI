"use client"

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react"
import type { PetEvent, PetSignal } from "@/components/pet"
import { DEFAULT_LISTENER_CONTROLS, MrtStream, type ListenerControls, type StreamState } from "@/lib/mrt-stream"

const IDLE_STATE: StreamState = {
  status: "idle", queuePosition: 0, listeners: 0, capacity: 0,
  message: null, bufferProgress: 0, variationPending: false,
}

function statusLabel(state: StreamState, wantsAudio: boolean): string {
  if (state.variationPending) return "finding a new take"
  switch (state.status) {
    case "loading": return state.message ?? "warming up the model"
    case "connecting": return state.message ?? "connecting"
    case "queued": return state.queuePosition > 0 ? `waiting for a slot · #${state.queuePosition}` : "waiting for a slot"
    case "buffering": return `catching up · ${Math.round(state.bufferProgress * 100)}%`
    case "live": return "live"
    case "paused": return "held"
    case "error": return state.message ?? "something broke"
    default: return wantsAudio ? "starting" : "ready"
  }
}

function useRadioState() {
  const [controls, setControls] = useState<ListenerControls>({ ...DEFAULT_LISTENER_CONTROLS })
  const [volume, setVolume] = useState(100)
  const [wantsAudio, setWantsAudio] = useState(false)
  const [streamState, setStreamState] = useState<StreamState>(IDLE_STATE)
  const [petSignal, setPetSignal] = useState<PetSignal | null>(null)
  const [focusMode, setFocusMode] = useState(false)
  const [sleepEndsAt, setSleepEndsAt] = useState<number | null>(null)
  const [sleepRemaining, setSleepRemaining] = useState(0)
  const streamRef = useRef<MrtStream | null>(null)
  const previousVolume = useRef(100)

  useEffect(() => {
    const stream = new MrtStream(setStreamState)
    streamRef.current = stream
    return () => {
      stream.destroy()
      streamRef.current = null
    }
  }, [])

  useEffect(() => { streamRef.current?.setControls(controls) }, [controls])
  useEffect(() => { streamRef.current?.setVolume(volume / 100) }, [volume])
  useEffect(() => {
    if (streamState.status === "error" && wantsAudio) setWantsAudio(false)
  }, [streamState.status, wantsAudio])

  const pausePlayback = useCallback(() => {
    streamRef.current?.pause()
    setWantsAudio(false)
  }, [])

  const togglePlayback = useCallback(async () => {
    if (!streamRef.current) return
    if (wantsAudio) { pausePlayback(); return }
    setWantsAudio(true)
    await streamRef.current.start(controls)
  }, [wantsAudio, controls, pausePlayback])

  const requestVariation = useCallback(() => { streamRef.current?.newVariation() }, [])
  const getLevel = useCallback(() => streamRef.current?.level() ?? 0, [])
  const getSpectrum = useCallback((out: Uint8Array) => streamRef.current?.spectrum(out) ?? 0, [])
  const handlePetEvent = useCallback((kind: PetEvent) => { setPetSignal({ kind, at: Date.now() }) }, [])
  const toggleMute = useCallback(() => {
    if (volume > 0) { previousVolume.current = volume; setVolume(0) }
    else setVolume(previousVolume.current || 60)
  }, [volume])

  const setSleepTimer = useCallback((minutes: number) => {
    setSleepEndsAt(minutes > 0 ? Date.now() + minutes * 60_000 : null)
    setSleepRemaining(minutes * 60)
  }, [])

  useEffect(() => {
    if (!sleepEndsAt) return
    const tick = () => {
      const left = Math.max(0, Math.ceil((sleepEndsAt - Date.now()) / 1000))
      setSleepRemaining(left)
      if (left === 0) { pausePlayback(); setSleepEndsAt(null) }
    }
    tick()
    const interval = window.setInterval(tick, 1000)
    return () => window.clearInterval(interval)
  }, [sleepEndsAt, pausePlayback])

  useEffect(() => {
    const handleKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement
      if (event.repeat || event.metaKey || event.ctrlKey || event.altKey ||
        target.closest("input, textarea, select, button, a, [contenteditable], dialog")) return
      if (event.code === "Space") { event.preventDefault(); void togglePlayback() }
      if (event.key.toLowerCase() === "m") toggleMute()
      if (event.key.toLowerCase() === "n" && wantsAudio && !streamState.variationPending) requestVariation()
    }
    window.addEventListener("keydown", handleKey)
    return () => window.removeEventListener("keydown", handleKey)
  }, [togglePlayback, toggleMute, requestVariation, wantsAudio, streamState.variationPending])

  const label = useMemo(() => statusLabel(streamState, wantsAudio), [streamState, wantsAudio])
  return {
    controls, setControls, volume, setVolume, wantsAudio, streamState,
    isLive: streamState.status === "live", petSignal, focusMode, setFocusMode,
    togglePlayback, requestVariation, getLevel, getSpectrum, handlePetEvent,
    toggleMute, label, sleepEndsAt, sleepRemaining, setSleepTimer,
  }
}

const RadioContext = createContext<ReturnType<typeof useRadioState> | null>(null)

export function RadioProvider({ children }: { children: ReactNode }) {
  const radio = useRadioState()
  return <RadioContext.Provider value={radio}>{children}</RadioContext.Provider>
}

export function useRadio() {
  const radio = useContext(RadioContext)
  if (!radio) throw new Error("useRadio must be used within RadioProvider")
  return radio
}
