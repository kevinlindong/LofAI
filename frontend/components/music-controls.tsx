import { useState, useEffect, RefObject } from "react"
import { Slider } from "@/components/ui/slider"
import { Button } from "@/components/ui/button"
import { Pause, Play } from "lucide-react"

interface MusicControlsProps {
  isPlaying: boolean
  togglePlayback: () => void
  mood: number
  setMood: (value: number) => void
  instrument: number
  setInstrument: (value: number) => void
  volume: number
  setVolume: (value: number) => void
  getMoodText: (value: number) => string
  getInstrumentType: (value: number) => string
  audioRef: RefObject<HTMLAudioElement | null>
}

export function MusicControls({
  isPlaying,
  togglePlayback,
  mood,
  setMood,
  instrument,
  setInstrument,
  volume,
  setVolume,
  getMoodText,
  getInstrumentType,
  audioRef,
}: MusicControlsProps) {
  const [isDraggingMood, setIsDraggingMood] = useState(false)
  const [isDraggingInstrument, setIsDraggingInstrument] = useState(false)
  const [playbackProgress, setPlaybackProgress] = useState(0)
  const [pulsePhase, setPulsePhase] = useState(0)

  useEffect(() => {
    let animationFrame: number

    const updateProgress = () => {
      if (audioRef.current && audioRef.current.duration) {
        // use actual audio time for progress
        const progress = audioRef.current.currentTime
        setPlaybackProgress(progress)
      }

      // smooth sine wave pulsing
      if (isPlaying) {
        const pulseSpeed = 2
        const phase = (Date.now() / 1000) * pulseSpeed
        const pulseAmount = Math.sin(phase) * 0.1 + 1
        setPulsePhase(pulseAmount)
      }

      if (isPlaying) {
        animationFrame = requestAnimationFrame(updateProgress)
      }
    }

    if (isPlaying) {
      animationFrame = requestAnimationFrame(updateProgress)
    } else {
      setPulsePhase(1)
    }

    return () => {
      if (animationFrame) {
        cancelAnimationFrame(animationFrame)
      }
    }
  }, [isPlaying, audioRef])

  const updatePrompt = async () => {
    const moodText = getMoodText(mood)
    const instrumentType = getInstrumentType(instrument)

    try {
      const response = await fetch("http://localhost:8000/api/update-prompt", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          mood: moodText,
          instruments: instrumentType,
        }),
      })
      
      if (response.ok) {
        console.log(`Updated prompt: mood=${moodText}, instrument=${instrumentType}`)
      }
    } catch (error) {
      console.error("Failed to update prompt:", error)
    }
  }

  const snapToValue = (value: number) => {
    const snapPoints = [
      { point: 0, range: [0, 25] },
      { point: 50, range: [25, 75] },
      { point: 100, range: [75, 100] },
    ]

    const snapPoint = snapPoints.find(({ range }) => value >= range[0] && value <= range[1])

    return snapPoint ? snapPoint.point : 50
  }

  const handleMoodDragStart = () => setIsDraggingMood(true)
  const handleMoodDragEnd = () => {
    setIsDraggingMood(false)
    const newValue = snapToValue(mood)
    setMood(newValue)
    updatePrompt()
  }

  const handleInstrumentDragStart = () => setIsDraggingInstrument(true)
  const handleInstrumentDragEnd = () => {
    setIsDraggingInstrument(false)
    const newValue = snapToValue(instrument)
    setInstrument(newValue)
    updatePrompt()
  }

  return (
    <div className="w-full h-full relative flex flex-col items-center justify-center">
      <div
        className="absolute inset-0 rounded-full"
        style={{
          background: `conic-gradient(from 0deg, hsl(var(--accent)) ${playbackProgress * 36}deg, transparent ${playbackProgress * 36}deg)`,
        }}
      />

      <div className="z-10 bg-card rounded-full w-[95%] h-[95%] flex flex-col items-center justify-center p-6">
        <div
          className="relative mb-12"
          style={{
            transform: `scale(${pulsePhase})`,
            transition: "transform 0.1s ease-in-out",
          }}
        >
          <Button
            size="lg"
            onClick={togglePlayback}
            className={`w-24 h-24 rounded-full relative z-10 transition-shadow duration-300 ${
              isPlaying ? "shadow-lg" : ""
            }`}
          >
            {isPlaying ? <Pause className="w-12 h-12" /> : <Play className="w-12 h-12" />}
          </Button>
        </div>

        <div className="w-full space-y-10">
          {/* mood slider */}
          <div className="w-full" style={{ padding: "10 0%" }}>
            <div
              className="flex justify-between text-xs dark:text-gray-300 font-medium mb-3"
              style={{
                width: "100%",
                marginLeft: "0%",
              }}
            >
              <span>Somber</span>
              <span>Neutral</span>
              <span>Lively</span>
            </div>
            <div className="relative">
              <Slider
                value={[mood]}
                onValueChange={(values) => setMood(values[0])}
                max={100}
                step={1}
                className="absolute inset-0"
                onPointerDown={handleMoodDragStart}
                onPointerUp={handleMoodDragEnd}
              />
            </div>
          </div>

          {/* instrument slider */}
          <div className="w-full" style={{ padding: "0 5%" }}>
            <div
              className="flex justify-between text-xs dark:text-gray-300 font-medium mb-3"
              style={{
                width: "100%",
                marginLeft: "0%",
              }}
            >
              <span>Piano</span>
              <span>Guitar</span>
              <span>Brass</span>
            </div>
            <div className="relative">
              <Slider
                value={[instrument]}
                onValueChange={(values) => setInstrument(values[0])}
                max={100}
                step={1}
                className="absolute inset-0"
                onPointerDown={handleInstrumentDragStart}
                onPointerUp={handleInstrumentDragEnd}
              />
            </div>
          </div>

          {/* volume slider */}
          <div className="w-full" style={{ padding: "0 11%" }}>
            <div
              className="flex justify-between text-xs dark:text-gray-300 font-medium mb-3"
              style={{
                width: "100%",
                marginLeft: "0%",
              }}
            >
              <span>0</span>
              <span>Volume</span>
              <span>100</span>
            </div>
            <div className="relative">
              <Slider
                value={[volume]}
                onValueChange={(values) => setVolume(values[0])}
                max={100}
                step={1}
                className="absolute inset-0"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default MusicControls