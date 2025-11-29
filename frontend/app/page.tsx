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
  const [volume, setVolume] = useState(100)
  const [currentTrack, setCurrentTrack] = useState(0)
  const [wsUrl, setWsUrl] = useState<string | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const isPlayingRef = useRef(false)

  // set websocket url after mount to avoid hydration error
  useEffect(() => {
    setWsUrl(`${window.location.protocol === "https:" ? "wss:" : "ws:"}//localhost:8000/ws`)
  }, [])

  useEffect(() => {
    if (!wsUrl) return

    try {
      wsRef.current = new WebSocket(wsUrl)
      
      wsRef.current.onerror = (error) => {
        console.log("WebSocket error:", error)
      }
      
      wsRef.current.onclose = () => {
        console.log("WebSocket closed")
      }
    } catch (error) {
      console.error("Failed to create WebSocket:", error)
    }

    return () => {
      wsRef.current?.close()
    }
  }, [wsUrl])

  // apply volume changes
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = volume / 100
    }
  }, [volume])

  // keep ref synced with state
  useEffect(() => {
    isPlayingRef.current = isPlaying
  }, [isPlaying])

  const loadAndPlayTrack = async (trackIndex: number) => {
    if (!audioRef.current) return
    
    try {
      console.log(`Loading track ${trackIndex}`)
      // load specific track
      audioRef.current.src = `http://localhost:8000/api/stream?track=${trackIndex}&t=${Date.now()}`
      audioRef.current.volume = volume / 100
      await audioRef.current.play()
    } catch (err) {
      console.error("Playback failed:", err)
    }
  }

  const advanceToNextTrack = async () => {
    try {
      // tell backend to advance
      const response = await fetch("http://localhost:8000/api/next-track", {
        method: "POST",
      })
      const data = await response.json()
      const nextTrack = data.track
      
      setCurrentTrack(nextTrack)
      
      // only play if still playing
      if (isPlayingRef.current) {
        await loadAndPlayTrack(nextTrack)
      }
    } catch (err) {
      console.error("Failed to advance track:", err)
    }
  }

  const togglePlayback = async () => {
    if (!audioRef.current) {
      audioRef.current = new Audio()
      audioRef.current.volume = volume / 100
      
      // advance to next when track ends
      audioRef.current.addEventListener("ended", () => {
        console.log("Track ended, advancing to next")
        advanceToNextTrack()
      })
    }

    if (isPlaying) {
      // pause without reloading
      audioRef.current.pause()
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send("paused")
      }
      setIsPlaying(false)
    } else {
      // play or resume
      try {
        // load track if no src set
        if (!audioRef.current.src || audioRef.current.src === '') {
          await loadAndPlayTrack(currentTrack)
        } else {
          // resume from pause
          await audioRef.current.play()
        }
        
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send("listening")
        }
        setIsPlaying(true)
      } catch (err) {
        console.error("Playback failed:", err)
      }
    }
  }

  const getMoodText = (value: number) => 
    value < 25 ? "somber" : value < 75 ? "neutral" : "lively"

  const getInstrumentType = (value: number) => 
    value < 25 ? "piano" : value < 75 ? "guitar" : "brass"

  return (
    <main className="min-h-screen bg-background text-foreground">
      <div className="relative min-h-screen p-4 sm:p-8 flex items-center justify-center">
        <ThemeToggle />
        <div className="relative w-full max-w-5xl flex flex-col sm:flex-row items-center justify-center gap-4 sm:gap-6">
          {/* music controls */}
          <div className="w-full sm:w-2/3 aspect-square rounded-full bg-card text-card-foreground shadow-lg p-4">
            <MusicControls
              isPlaying={isPlaying}
              togglePlayback={togglePlayback}
              mood={mood}
              setMood={setMood}
              instrument={instrument}
              setInstrument={setInstrument}
              volume={volume}
              setVolume={setVolume}
              getMoodText={getMoodText}
              getInstrumentType={getInstrumentType}
              audioRef={audioRef}
            />
          </div>

          <div className="flex flex-row sm:flex-col gap-4 sm:gap-6 w-full sm:w-1/3">
            {/* pomodoro timer */}
            <div className="w-1/2 sm:w-full aspect-square rounded-full bg-card text-card-foreground shadow-lg p-3">
              <PomodoroTimer />
            </div>

            {/* todo list */}
            <div className="w-1/2 sm:w-full aspect-square rounded-full bg-card text-card-foreground shadow-lg p-3">
              <TodoList />
            </div>
          </div>
        </div>
      </div>
    </main>
  )
}