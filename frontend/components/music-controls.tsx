"use client"

import { DotGlyph } from "@/components/dot-glyph"
import { DotSlider } from "@/components/dot-slider"
import { DotVisualizer } from "@/components/dot-visualizer"

interface MusicControlsProps {
  isPlaying: boolean
  togglePlayback: () => void
  mood: number
  setMood: (value: number) => void
  instrument: number
  setInstrument: (value: number) => void
  volume: number
  setVolume: (value: number) => void
  statusLabel: string
  moodText: string
  instrumentText: string
  isLive: boolean
  getSpectrum: (out: Uint8Array) => number
}

// mood and instrument are three settings each, not a continuum - the slider is
// how you travel between them, and it settles on the nearest one when let go
const snap = (value: number) => (value < 25 ? 0 : value < 75 ? 50 : 100)

export function MusicControls({
  isPlaying,
  togglePlayback,
  mood,
  setMood,
  instrument,
  setInstrument,
  volume,
  setVolume,
  statusLabel,
  moodText,
  instrumentText,
  isLive,
  getSpectrum,
}: MusicControlsProps) {
  return (
    <div className="flex h-full flex-col items-center justify-between gap-8 p-6 sm:p-8">
      <div className="flex flex-1 items-center justify-center w-full min-h-0">
        <DotVisualizer getSpectrum={getSpectrum} active={isLive}>
          <div className="flex flex-col items-center gap-4">
            <button
              type="button"
              onClick={togglePlayback}
              aria-label={isPlaying ? "Pause" : "Play"}
              aria-pressed={isPlaying}
              className="key h-[4.5rem] w-[4.5rem] rounded-full"
            >
              <DotGlyph name={isPlaying ? "pause" : "play"} dot={4} />
            </button>

            <span className="label h-3 text-center whitespace-nowrap">
              {statusLabel}
            </span>
          </div>
        </DotVisualizer>
      </div>

      <div className="w-full max-w-sm space-y-6">
        <DotSlider
          label="Mood"
          readout={moodText}
          value={mood}
          onChange={setMood}
          onRelease={() => setMood(snap(mood))}
        />
        <DotSlider
          label="Voice"
          readout={instrumentText}
          value={instrument}
          onChange={setInstrument}
          onRelease={() => setInstrument(snap(instrument))}
        />
        <DotSlider label="Volume" readout={`${volume}`} value={volume} onChange={setVolume} />
      </div>
    </div>
  )
}

export default MusicControls
