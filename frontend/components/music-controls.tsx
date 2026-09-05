"use client"

import { DotGlyph } from "@/components/dot-glyph"
import { DotSlider } from "@/components/dot-slider"
import { DotVisualizer } from "@/components/dot-visualizer"
import type { ListenerControls } from "@/lib/mrt-stream"

interface StationPreset {
  id: string
  label: string
  description: string
  controls: Omit<ListenerControls, "station">
}

export const STATION_PRESETS: readonly StationPreset[] = [
  {
    id: "dusty-beats",
    label: "Dusty beats",
    description: "Warm guitar loops and an easy pocket",
    controls: {
      mood: "neutral",
      instrument: "guitar",
      bpm: 76,
      groove: 0.62,
      intensity: 0.42,
      melody: true,
      drums: true,
    },
  },
  {
    id: "rainy-piano",
    label: "Rainy piano",
    description: "Spacious felt piano for quiet focus",
    controls: {
      mood: "somber",
      instrument: "piano",
      bpm: 68,
      groove: 0.4,
      intensity: 0.28,
      melody: true,
      drums: true,
    },
  },
  {
    id: "jazz-cafe",
    label: "Jazz cafe",
    description: "Warm jazz guitar with a loose brushed swing",
    controls: {
      mood: "neutral",
      instrument: "guitar",
      bpm: 82,
      groove: 0.7,
      intensity: 0.52,
      melody: true,
      drums: true,
    },
  },
  {
    id: "sunlit-groove",
    label: "Sunlit groove",
    description: "Muted brass and a brighter, animated beat",
    controls: {
      mood: "lively",
      instrument: "brass",
      bpm: 94,
      groove: 0.78,
      intensity: 0.68,
      melody: true,
      drums: true,
    },
  },
] as const

interface MusicControlsProps {
  isPlaying: boolean
  togglePlayback: () => void
  requestVariation: () => void
  variationPending: boolean
  controls: ListenerControls
  setControls: (controls: ListenerControls) => void
  volume: number
  setVolume: (value: number) => void
  statusLabel: string
  isLive: boolean
  getSpectrum: (out: Uint8Array) => number
}

// mood and instrument are three settings each, not a continuum - the slider is
// how you travel between them, and it settles on the nearest one when let go
const snap = (value: number) => (value < 25 ? 0 : value < 75 ? 50 : 100)

const moodValue = (mood: string) => (mood === "somber" ? 0 : mood === "lively" ? 100 : 50)
const instrumentValue = (instrument: string) =>
  instrument === "piano" ? 0 : instrument === "brass" ? 100 : 50
const moodFor = (value: number) => (value < 25 ? "somber" : value < 75 ? "neutral" : "lively")
const instrumentFor = (value: number) =>
  value < 25 ? "piano" : value < 75 ? "guitar" : "brass"

function rangeLabel(value: number, low: string, middle: string, high: string) {
  if (value < 0.34) return low
  if (value < 0.67) return middle
  return high
}

export function MusicControls({
  isPlaying,
  togglePlayback,
  requestVariation,
  variationPending,
  controls,
  setControls,
  volume,
  setVolume,
  statusLabel,
  isLive,
  getSpectrum,
}: MusicControlsProps) {
  const selectedStation = STATION_PRESETS.find((station) => station.id === controls.station)
  const update = (next: Partial<ListenerControls>, customizeStyle = false) => {
    setControls({
      ...controls,
      ...next,
      station: customizeStyle ? "custom" : controls.station,
    })
  }

  const chooseStation = (id: string) => {
    const preset = STATION_PRESETS.find((station) => station.id === id)
    if (preset) setControls({ station: preset.id, ...preset.controls })
  }

  return (
    <div className="music-control-layout">
      <div className="visualizer-stage flex-col gap-3">
        <DotVisualizer getSpectrum={getSpectrum} active={isLive}>
          <div className="flex flex-col items-center gap-4">
            <button
              type="button"
              onClick={togglePlayback}
              aria-label={isPlaying ? "Pause" : "Play"}
              aria-pressed={isPlaying}
              className="key play-key"
            >
              <DotGlyph name={isPlaying ? "pause" : "play"} dot={4} />
            </button>

            <span className="play-status">{statusLabel}</span>
          </div>
        </DotVisualizer>

        <button
          type="button"
          onClick={requestVariation}
          disabled={!isPlaying || variationPending}
          className="key relative z-10 inline-flex min-h-[2.5rem] items-center gap-3 rounded-full px-4 text-xs disabled:cursor-not-allowed disabled:opacity-40"
          aria-label="Skip to a new music variation"
        >
          <DotGlyph name="rewind" dot={2} />
          <span>{variationPending ? "finding a new take" : "new take"}</span>
        </button>
      </div>

      <div className="control-bank">
        <div className="control-station">
          <div className="mb-2 flex items-baseline justify-between">
            <label htmlFor="music-station" className="label">
              Station
            </label>
            <span className="readout text-xs">{selectedStation?.label ?? "custom mix"}</span>
          </div>
          <select
            id="music-station"
            value={controls.station}
            onChange={(event) => chooseStation(event.target.value)}
            className="w-full rounded-md border bg-transparent px-3 py-2 text-sm"
            style={{ borderColor: "var(--line-color)", color: "var(--text)" }}
          >
            {controls.station === "custom" && <option value="custom">Custom mix</option>}
            {STATION_PRESETS.map((station) => (
              <option key={station.id} value={station.id}>
                {station.label}
              </option>
            ))}
          </select>
          <p className="mt-2 text-xs" style={{ color: "var(--text-faint)" }}>
            {selectedStation?.description ?? "Your own balance of mood, rhythm, and arrangement"}
          </p>
        </div>

        <DotSlider
          label="Mood"
          readout={controls.mood}
          value={moodValue(controls.mood)}
          onChange={(value) => update({ mood: moodFor(value) }, true)}
          onRelease={() => update({ mood: moodFor(snap(moodValue(controls.mood))) }, true)}
        />
        <DotSlider
          label="Voice"
          readout={controls.instrument}
          value={instrumentValue(controls.instrument)}
          onChange={(value) => update({ instrument: instrumentFor(value) }, true)}
          onRelease={() =>
            update(
              { instrument: instrumentFor(snap(instrumentValue(controls.instrument))) },
              true,
            )
          }
        />
        <DotSlider
          label="Tempo"
          readout={`${controls.bpm} bpm`}
          value={controls.bpm}
          min={60}
          max={110}
          step={1}
          onChange={(bpm) => update({ bpm })}
        />
        <DotSlider
          label="Groove"
          readout={rangeLabel(controls.groove, "straight", "laid-back", "loose")}
          value={Math.round(controls.groove * 100)}
          onChange={(groove) => update({ groove: groove / 100 })}
        />
        <DotSlider
          label="Energy"
          readout={rangeLabel(controls.intensity, "hushed", "steady", "bright")}
          value={Math.round(controls.intensity * 100)}
          onChange={(intensity) => update({ intensity: intensity / 100 })}
        />
        <DotSlider label="Volume" readout={`${volume}`} value={volume} onChange={setVolume} />

        <div
          className="control-arrangement grid grid-cols-2 gap-3"
          role="group"
          aria-label="Arrangement layers"
        >
          <button
            type="button"
            aria-pressed={controls.melody}
            title="Turn off the composed note guide and let the model improvise pitches"
            onClick={() => update({ melody: !controls.melody })}
            className="key min-h-[2.5rem] rounded-md px-3 text-xs"
            style={{ opacity: controls.melody ? 1 : 0.55 }}
          >
            music guide {controls.melody ? "on" : "off"}
          </button>
          <button
            type="button"
            aria-pressed={controls.drums}
            onClick={() => update({ drums: !controls.drums })}
            className="key min-h-[2.5rem] rounded-md px-3 text-xs"
            style={{ opacity: controls.drums ? 1 : 0.55 }}
          >
            drums {controls.drums ? "on" : "off"}
          </button>
        </div>
      </div>
    </div>
  )
}

export default MusicControls
