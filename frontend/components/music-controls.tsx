"use client"

import { DotGlyph } from "@/components/dot-glyph"
import { DotSlider } from "@/components/dot-slider"
import { DotVisualizer } from "@/components/dot-visualizer"
import { SoundEditor } from "@/components/sound-editor"
import type { RadioControls } from "@/lib/sound-recipe"

interface MusicControlsProps {
  isPlaying: boolean
  togglePlayback: () => void
  requestVariation: () => void
  variationPending: boolean
  controls: RadioControls
  setControls: (controls: RadioControls) => void
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
  const update = (next: Partial<RadioControls>) => setControls({ ...controls, ...next })

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
        <SoundEditor />
        <div className="control-playback" role="group" aria-label="Music options">
          <DotSlider label="Volume" readout={`${volume}%`} value={volume} onChange={setVolume} />
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
