"use client"

import { DotGlyph } from "@/components/dot-glyph"
import { DotSlider } from "@/components/dot-slider"
import { DotVisualizer } from "@/components/dot-visualizer"
import type { ListenerControls } from "@/lib/mrt-stream"

interface StationPreset {
  id: string
  label: string
  description: string
}

export const STATION_PRESETS: readonly StationPreset[] = [
  {
    id: "dusty-beats",
    label: "Dusty beats",
    description: "Warm guitar loops and an easy pocket",
  },
  {
    id: "rainy-piano",
    label: "Rainy piano",
    description: "Spacious felt piano for quiet focus",
  },
  {
    id: "jazz-cafe",
    label: "Jazz cafe",
    description: "Warm jazz guitar with a loose brushed swing",
  },
  {
    id: "sunlit-groove",
    label: "Sunlit groove",
    description: "Muted brass and a brighter, animated beat",
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
  const update = (next: Partial<ListenerControls>) => setControls({ ...controls, ...next })

  const chooseStation = (id: string) => {
    const preset = STATION_PRESETS.find((station) => station.id === id)
    if (preset) update({ station: preset.id })
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
            {STATION_PRESETS.map((station) => (
              <option key={station.id} value={station.id}>
                {station.label}
              </option>
            ))}
          </select>
          <p className="mt-2 text-xs" style={{ color: "var(--text-faint)" }}>
            {selectedStation?.description ?? "Choose a station"}
          </p>
        </div>

        <DotSlider label="Volume" readout={`${volume}`} value={volume} onChange={setVolume} />

        <div
          className="control-arrangement"
          role="group"
          aria-label="Music options"
        >
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
