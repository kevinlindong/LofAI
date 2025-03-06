"use client"

import { useEffect, useRef, useState } from "react"
import { MusicControls } from "@/components/music-controls"
import { PomodoroTimer } from "@/components/pomodoro-timer"
import { TodoList } from "@/components/todo-list"
import { ThemeToggle } from "@/components/theme-toggle"

export default function LofiGenerator() {
  const [isPlaying, setIsPlaying] = useState(false)
  const [mood, setMood] = useState(50)
  const [instrument, setInstrument] = useState(50)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    wsRef.current = new WebSocket(
      `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/ws`,
    )

    return () => {
      wsRef.current?.close()
    }
  }, [])

  const togglePlayback = async () => {
    if (!audioRef.current) {
      audioRef.current = new Audio("/api/stream")
    }

    if (isPlaying) {
      audioRef.current.pause()
      wsRef.current?.send("paused")
    } else {
      try {
        await audioRef.current.play()
        wsRef.current?.send("listening")
      } catch (err) {
        console.error("Playback failed:", err)
      }
    }

    setIsPlaying(!isPlaying)
  }

  const getMoodText = (value: number) => {
    if (value < 25) return "somber"
    if (value < 75) return "neutral"
    return "lively"
  }

  const getInstrumentType = (value: number) => {
    if (value < 25) return "piano"
    if (value < 75) return "guitar"
    return "brass"
  }

  return (
    <main className="min-h-screen bg-background text-foreground">
      <div className="relative min-h-screen p-4 sm:p-8 flex items-center justify-center">
        <ThemeToggle />
        <div className="relative w-full max-w-5xl flex flex-col sm:flex-row items-center justify-center gap-4 sm:gap-6">
          {/* Large circle for music controls */}
          <div className="w-full sm:w-2/3 aspect-square rounded-full bg-card text-card-foreground shadow-lg p-4">
            <MusicControls
              isPlaying={isPlaying}
              togglePlayback={togglePlayback}
              mood={mood}
              setMood={setMood}
              instrument={instrument}
              setInstrument={setInstrument}
              getMoodText={getMoodText}
              getInstrumentType={getInstrumentType}
            />
          </div>

          <div className="flex flex-row sm:flex-col gap-4 sm:gap-6 w-full sm:w-1/3">
            {/* Smaller circle for Pomodoro Timer */}
            <div className="w-1/2 sm:w-full aspect-square rounded-full bg-card text-card-foreground shadow-lg p-3">
              <PomodoroTimer />
            </div>

            {/* Smaller circle for Todo List */}
            <div className="w-1/2 sm:w-full aspect-square rounded-full bg-card text-card-foreground shadow-lg p-3">
              <TodoList />
            </div>
          </div>
        </div>
      </div>
    </main>
  )
}