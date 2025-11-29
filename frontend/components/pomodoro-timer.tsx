import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Play, Pause, RotateCcw } from "lucide-react"

export function PomodoroTimer() {
  const [isRunning, setIsRunning] = useState(false)
  const [timeLeft, setTimeLeft] = useState(25 * 60)
  const [isBreak, setIsBreak] = useState(false)
  const [workDuration, setWorkDuration] = useState(25)
  const [breakDuration, setBreakDuration] = useState(5)
  const [progress, setProgress] = useState(0)
  const timerRef = useRef<NodeJS.Timeout | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const progressRef = useRef<number>(0)
  const lastTickRef = useRef<number>(0)

  useEffect(() => {
    // Initialize Audio after mount to prevent hydration errors
    if (typeof window !== "undefined") {
      audioRef.current = new Audio("/timer-end.mp3")
      lastTickRef.current = Date.now()
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [])

  const startTimer = () => {
    setIsRunning(true)
    lastTickRef.current = Date.now()
    timerRef.current = setInterval(() => {
      const now = Date.now()
      const deltaTime = (now - lastTickRef.current) / 1000
      lastTickRef.current = now

      setTimeLeft((prev) => {
        if (prev <= deltaTime) {
          audioRef.current?.play()
          if (timerRef.current) clearInterval(timerRef.current)
          setIsRunning(false)
          const nextIsBreak = !isBreak
          setIsBreak(nextIsBreak)
          return nextIsBreak ? breakDuration * 60 : workDuration * 60
        }
        return prev - deltaTime
      })
    }, 50) // Update more frequently for smoother animation
  }

  const pauseTimer = () => {
    setIsRunning(false)
    if (timerRef.current) clearInterval(timerRef.current)
  }

  const resetTimer = () => {
    setIsRunning(false)
    if (timerRef.current) clearInterval(timerRef.current)
    setTimeLeft(workDuration * 60)
    setIsBreak(false)
    setProgress(0)
    progressRef.current = 0
  }

  // Smooth progress update
  useEffect(() => {
    let animationFrame: number

    const updateProgress = () => {
      if (isRunning) {
        const duration = isBreak ? breakDuration * 60 : workDuration * 60
        const timeElapsed = isBreak ? 
          (breakDuration * 60 - timeLeft) : 
          (workDuration * 60 - timeLeft)
        
        const targetProgress = isBreak ? 
          1 - (timeElapsed / duration) : 
          timeElapsed / duration

        // Smooth progress interpolation
        const diff = targetProgress - progressRef.current
        const step = diff * 0.1 // Adjust this value to control smoothness
        progressRef.current += step
        setProgress(progressRef.current)
      }

      animationFrame = requestAnimationFrame(updateProgress)
    }

    animationFrame = requestAnimationFrame(updateProgress)

    return () => {
      cancelAnimationFrame(animationFrame)
    }
  }, [isRunning, timeLeft, isBreak, workDuration, breakDuration])

  // Reset progress when switching between work and break
  useEffect(() => {
    progressRef.current = isBreak ? 1 : 0
    setProgress(progressRef.current)
  }, [isBreak])

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`
  }

  return (
    <div className="w-full max-w-md mx-auto aspect-square flex flex-col items-center justify-center relative">
      <div
        className="absolute inset-0 rounded-full"
        style={{
          background: isBreak
            ? `conic-gradient(from 0deg, transparent ${(1 - progress) * 360}deg, hsl(var(--accent)) ${(1 - progress) * 360}deg)`
            : `conic-gradient(from 0deg, hsl(var(--accent)) ${progress * 360}deg, transparent ${progress * 360}deg)`,
          transition: 'background 0.1s linear',
        }}
      />

      <div className="z-10 bg-card rounded-full w-[95%] h-[95%] flex flex-col items-center justify-center">
        <div className="text-base xs:text-lg sm:text-2xl md:text-3xl font-bold mb-1 sm:mb-2 dark:text-white">
          {formatTime(timeLeft)}
        </div>

        <div className="flex space-x-1 xs:space-x-2 mb-2 xs:mb-4">
          <Button
            onClick={isRunning ? pauseTimer : startTimer}
            variant="outline"
            size="sm"
            className="h-5 w-5 xs:h-6 xs:w-6 sm:h-8 sm:w-8 p-0"
          >
            {isRunning ? (
              <Pause className="h-2 w-2 xs:h-3 xs:w-3 sm:h-4 sm:w-4" />
            ) : (
              <Play className="h-2 w-2 xs:h-3 xs:w-3 sm:h-4 sm:w-4" />
            )}
          </Button>
          <Button 
            onClick={resetTimer} 
            variant="outline" 
            size="sm" 
            className="h-5 w-5 xs:h-6 xs:w-6 sm:h-8 sm:w-8 p-0"
          >
            <RotateCcw className="h-2 w-2 xs:h-3 xs:w-3 sm:h-4 sm:w-4" />
          </Button>
        </div>

        <div className="w-full space-y-2 xs:space-y-4 sm:space-y-7 px-2 sm:px-4">
          {[
            { label: "Work", value: workDuration, setValue: setWorkDuration, max: 60 },
            { label: "Break", value: breakDuration, setValue: setBreakDuration, max: 30 },
          ].map((item, index) => (
            <div key={item.label} className="space-y-2 xs:space-y-3">
              <div
                className="flex justify-between text-[8px] xs:text-[10px] sm:text-xs dark:text-gray-300 font-medium"
                style={{
                  width: `${100 - (index + 1.5) * 10}%`,
                  marginLeft: `${(index + 1.5) * 5}%`,
                }}
              >
                <span>{item.label}</span>
                <span>{item.value} min</span>
              </div>
              <div className="relative">
                <Slider
                  value={[item.value]}
                  onValueChange={(values) => {
                    item.setValue(values[0])
                    if (!isRunning && (index === 0 ? !isBreak : isBreak)) {
                      setTimeLeft(values[0] * 60)
                    }
                  }}
                  min={1} 
                  max={item.max}
                  step={1}
                  disabled={isRunning}
                  className="absolute inset-0 [&_[role='slider']]:h-2 [&_[role='slider']]:w-2 xs:[&_[role='slider']]:h-3 xs:[&_[role='slider']]:w-3 sm:[&_[role='slider']]:h-4 sm:[&_[role='slider']]:w-4"
                  style={{
                    width: `${100 - (index + 1.5) * 10}%`,
                    left: `${(index + 1.5) * 5}%`,
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}