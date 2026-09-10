"use client"

import { DotGlyph } from "@/components/dot-glyph"
import { DotSlider } from "@/components/dot-slider"
import { DotVisualizer } from "@/components/dot-visualizer"
import { MAX_CUSTOM_PROMPT_CHARS, type ListenerControls } from "@/lib/mrt-stream"

interface StationPreset {
  id: string
  label: string
  description: string
}

export const CUSTOM_STATION = "custom"

// Named stations, without the custom-prompt entry. The alternate designs use
// this list because they present stations as fixed tiles and have no free-text
// prompt field of their own.
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

// The original interface adds a "Custom prompt" option to the named stations.
export const STATION_OPTIONS: readonly StationPreset[] = [
  ...STATION_PRESETS,
  {
    id: CUSTOM_STATION,
    label: "Custom prompt",
    description: "Describe your own lofi — anything you like",
  },
] as const

// A few ideas so the empty field is not intimidating. All stay in the lofi
// lane the backend scaffolds around.
const PROMPT_SUGGESTIONS = [
  "rainy tokyo rooftop, muted saxophone",
  "sleepy vinyl piano, soft rain",
  "8-bit chiptune lofi, gentle beat",
  "bossa nova guitar, warm tape hiss",
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

// The two granular dials are stored 0..1 but presented as friendly 0..100
// percentages on the same rail vocabulary as volume.
const toPercent = (value: number) => Math.round(value * 100)
const fromPercent = (value: number) => value / 100

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
  const isCustom = controls.station === CUSTOM_STATION
  const selectedStation = STATION_OPTIONS.find((station) => station.id === controls.station)
  const update = (next: Partial<ListenerControls>) => setControls({ ...controls, ...next })

  const chooseStation = (id: string) => {
    const preset = STATION_OPTIONS.find((station) => station.id === id)
    if (!preset) return
    // Switching to a named station drops any stale custom prompt; the backend
    // does the same, but clearing it here keeps the field and UI honest.
    if (preset.id === CUSTOM_STATION) update({ station: CUSTOM_STATION })
    else update({ station: preset.id, customPrompt: "" })
  }

  const setPrompt = (text: string) =>
    update({ station: CUSTOM_STATION, customPrompt: text.slice(0, MAX_CUSTOM_PROMPT_CHARS) })

  return (
    <div className="music-control-layout">
      <div className="visualizer-stage flex-col gap-3">
        <DotVisualizer getSpectrum={getSpectrum} active={isLive}>
          <div className="flex flex-col items-center gap-3">
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
          className="key relative z-10 inline-flex min-h-[2.25rem] items-center gap-3 rounded-full px-4 text-xs disabled:cursor-not-allowed disabled:opacity-40"
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
            className="control-select"
          >
            {STATION_OPTIONS.map((station) => (
              <option key={station.id} value={station.id}>
                {station.label}
              </option>
            ))}
          </select>

          {isCustom ? (
            <div className="control-prompt">
              <input
                id="music-prompt"
                type="text"
                value={controls.customPrompt}
                maxLength={MAX_CUSTOM_PROMPT_CHARS}
                placeholder="e.g. rainy tokyo rooftop, muted saxophone"
                onChange={(event) => setPrompt(event.target.value)}
                className="control-prompt-input"
                aria-label="Describe the music you want to hear"
                autoComplete="off"
                spellCheck={false}
              />
              <div className="control-prompt-suggestions" aria-hidden={!isCustom}>
                {PROMPT_SUGGESTIONS.map((idea) => (
                  <button
                    key={idea}
                    type="button"
                    className="prompt-chip"
                    onClick={() => setPrompt(idea)}
                  >
                    {idea}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <p className="mt-2 text-xs" style={{ color: "var(--text-faint)" }}>
              {selectedStation?.description ?? "Choose a station"}
            </p>
          )}
        </div>

        <div className="control-dials">
          <DotSlider
            label="Style match"
            readout={`${toPercent(controls.adherence)}`}
            value={toPercent(controls.adherence)}
            onChange={(value) => update({ adherence: fromPercent(value) })}
          />
          <DotSlider
            label="Variation"
            readout={`${toPercent(controls.variation)}`}
            value={toPercent(controls.variation)}
            onChange={(value) => update({ variation: fromPercent(value) })}
          />
        </div>

        <DotSlider label="Volume" readout={`${volume}`} value={volume} onChange={setVolume} />

        <div className="control-arrangement" role="group" aria-label="Music options">
          <button
            type="button"
            aria-pressed={controls.drums}
            onClick={() => update({ drums: !controls.drums })}
            className="key min-h-[2.25rem] rounded-md px-3 text-xs"
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
