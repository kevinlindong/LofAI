"use client"

import { useId, useState, type KeyboardEvent } from "react"
import { DotGlyph, type GlyphName } from "@/components/dot-glyph"
import { DotSlider } from "@/components/dot-slider"
import { useRadio } from "@/components/radio-provider"
import { MAX_CUSTOM_PROMPT_CHARS } from "@/lib/mrt-stream"
import {
  buildSoundPrompt, CUSTOM_STATION, EFFECTS, INSTRUMENTS, MAX_EFFECTS, MAX_INSTRUMENTS,
  MOODS, sameRecipe, STATION_PRESETS, VIBES, type SoundOption,
} from "@/lib/sound-recipe"

const PRESET_ICONS: GlyphName[] = ["music", "moon", "music", "sun"]
const CATEGORIES = ["Instruments", "Vibes", "Moods", "Effects"] as const
const PROMPT_IDEAS = [
  { label: "Rainy rooftop", prompt: "rainy Tokyo rooftop, mellow saxophone, soft piano, distant rain" },
  { label: "Slow Sunday", prompt: "lazy Sunday morning, warm jazz guitar, Rhodes keys, gentle tape warmth" },
  { label: "Midnight arcade", prompt: "dreamy chiptune lo-fi, music box, soft synth pads, sleepy midnight mood" },
]

// Both tab rows use the same native button focus and keyboard behavior.
function moveTab(event: KeyboardEvent<HTMLButtonElement>) {
  if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return
  const tabs = Array.from(event.currentTarget.parentElement!.querySelectorAll<HTMLButtonElement>('[role="tab"]'))
  const index = tabs.indexOf(event.currentTarget)
  const next = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 :
    (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length
  event.preventDefault()
  tabs[next].click()
  tabs[next].focus()
}

function SingleChoice({ options, value, onChange, label, name }: {
  options: readonly SoundOption[]; value: string; onChange: (value: string) => void; label: string; name: string
}) {
  return (
    <div className="sound-chips" role="radiogroup" aria-label={label}>
      {options.map((option) => (
        <label key={option.id} className={`sound-chip ${value === option.id ? "is-chosen" : ""}`}>
          <input className="sr-only" type="radio" name={name} value={option.id} checked={value === option.id} onChange={() => onChange(option.id)} />
          <span className="sound-chip-mark" aria-hidden="true">{value === option.id ? <DotGlyph name="check" dot={1} /> : <span />}</span>
          {option.label}
        </label>
      ))}
    </div>
  )
}

export function SoundEditor({ showPresets = true }: { showPresets?: boolean }) {
  const { controls, setControls, soundDraft, setSoundDraft, selectStation, wantsAudio } = useRadio()
  const id = useId()
  const [category, setCategory] = useState<(typeof CATEGORIES)[number]>("Instruments")
  const { mode, recipe, prompt } = soundDraft
  const generatedPrompt = buildSoundPrompt(recipe)
  const nextPrompt = mode === "builder" ? generatedPrompt : prompt.replace(/\s+/g, " ").trim()
  const namedPreset = STATION_PRESETS.find((preset) => preset.id === controls.station)
  const currentRecipe = controls.recipe ?? namedPreset?.recipe
  const isCurrent = mode === "builder"
    ? !!currentRecipe && sameRecipe(recipe, currentRecipe)
    : nextPrompt.length > 0 && controls.station === CUSTOM_STATION && !controls.recipe && nextPrompt === controls.customPrompt.trim()
  const canApply = !isCurrent && nextPrompt.length > 0 && nextPrompt.length <= MAX_CUSTOM_PROMPT_CHARS

  const updateRecipe = (next: Partial<typeof recipe>) =>
    setSoundDraft((draft) => ({ ...draft, recipe: { ...draft.recipe, ...next } }))

  const toggleChoice = (key: "instruments" | "effects", value: string) => {
    const selected = recipe[key]
    const next = selected.includes(value) ? selected.filter((entry) => entry !== value) : [...selected, value]
    updateRecipe({ [key]: next })
  }

  const applySound = () => {
    if (!canApply) return
    const preset = mode === "builder" && STATION_PRESETS.find((entry) => sameRecipe(entry.recipe, recipe))
    if (preset) selectStation(preset.id)
    else setControls({ ...controls, station: CUSTOM_STATION, customPrompt: nextPrompt, recipe: mode === "builder" ? recipe : undefined })
  }

  const status = isCurrent
    ? wantsAudio ? "Your sound is set. Let it settle in." : "Sound set. Press play when you’re ready."
    : !nextPrompt ? "A few words are all it takes."
    : nextPrompt.length > MAX_CUSTOM_PROMPT_CHARS ? "Shorten your mix to apply it." : "Changes ready when you are."

  return (
    <section className="sound-editor" aria-labelledby={`${id}-title`}>
      <div className="sound-heading">
        <div><p className="sound-eyebrow">A little more you</p><h2 id={`${id}-title`}>Shape your sound.</h2></div>
        <span className="sound-heading-glyph"><DotGlyph name="settings" dot={2} /></span>
      </div>

      <div className="sound-mode-tabs" role="tablist" aria-label="How to shape your sound">
        {([ ["builder", "Build a vibe", "settings"], ["prompt", "Write a prompt", "music"] ] as const).map(([value, label, glyph]) => (
          <button key={value} type="button" role="tab" id={`${id}-${value}-tab`} aria-selected={mode === value} aria-controls={`${id}-${value}-panel`} tabIndex={mode === value ? 0 : -1} onKeyDown={moveTab} onClick={() => setSoundDraft((draft) => ({ ...draft, mode: value }))}>
            <DotGlyph name={glyph} dot={1} /><span>{label}</span>
          </button>
        ))}
      </div>

      <div role="tabpanel" id={`${id}-builder-panel`} aria-labelledby={`${id}-builder-tab`} tabIndex={0} className="sound-mode-panel" hidden={mode !== "builder"}>
            {showPresets && (
              <div className="sound-presets" role="group" aria-label="Preset vibes">
                {STATION_PRESETS.map((preset, index) => (
                  <button key={preset.id} type="button" className="sound-preset" aria-pressed={controls.station === preset.id && sameRecipe(recipe, preset.recipe)} onClick={() => selectStation(preset.id)}>
                    <span className="sound-preset-icon"><DotGlyph name={PRESET_ICONS[index]} dot={2} /></span>
                    <span><strong>{preset.label}</strong><small>{index === 0 ? "Warm / easy" : index === 1 ? "Quiet / spacious" : index === 2 ? "Loose / soulful" : "Bright / breezy"}</small></span>
                    <span className="sound-preset-dot" aria-hidden="true" />
                  </button>
                ))}
              </div>
            )}

            <div className="sound-detail-tabs" role="tablist" aria-label="Customize your sound">
              {CATEGORIES.map((value) => (
                <button key={value} type="button" role="tab" id={`${id}-${value}-tab`} aria-controls={`${id}-${value}-panel`} aria-selected={category === value} tabIndex={category === value ? 0 : -1} onKeyDown={moveTab} onClick={() => setCategory(value)}>
                  {value}{(value === "Instruments" || value === "Effects") && <span>{value === "Instruments" ? recipe.instruments.length : recipe.effects.length}</span>}
                </button>
              ))}
            </div>

            {CATEGORIES.map((panel) => (
            <div key={panel} className="sound-choice-panel" role="tabpanel" id={`${id}-${panel}-panel`} aria-labelledby={`${id}-${panel}-tab`} tabIndex={0} hidden={category !== panel}>
              <p className="sound-hint">
                {panel === "Instruments" && `Pick your players. Choose 1–${MAX_INSTRUMENTS}.`}
                {panel === "Vibes" && "The style that ties it all together. Choose one."}
                {panel === "Moods" && "How should it feel? Choose one."}
                {panel === "Effects" && `Suggest a little texture. Choose up to ${MAX_EFFECTS}, or leave it bare.`}
              </p>
              {panel === "Vibes" && <SingleChoice options={VIBES} value={recipe.vibe} onChange={(vibe) => updateRecipe({ vibe })} label="Vibe" name={`${id}-vibe`} />}
              {panel === "Moods" && <SingleChoice options={MOODS} value={recipe.mood} onChange={(mood) => updateRecipe({ mood })} label="Mood" name={`${id}-mood`} />}
              {(panel === "Instruments" || panel === "Effects") && (
                <div className="sound-chips" role="group" aria-label={panel}>
                  {(panel === "Instruments" ? INSTRUMENTS : EFFECTS).map((option) => {
                    const key = panel === "Instruments" ? "instruments" : "effects"
                    const selected = recipe[key].includes(option.id)
                    const atLimit = recipe[key].length >= (key === "instruments" ? MAX_INSTRUMENTS : MAX_EFFECTS)
                    return (
                      <button key={option.id} type="button" className={`sound-chip ${selected ? "is-chosen" : ""}`} aria-pressed={selected} disabled={(!selected && atLimit) || (selected && key === "instruments" && recipe.instruments.length === 1)} onClick={() => toggleChoice(key, option.id)}>
                        <span className="sound-chip-mark"><DotGlyph name={selected ? "check" : "plus"} dot={1} /></span>{option.label}
                      </button>
                    )
                  })}
                </div>
              )}
            </div>
            ))}
            <div className="sound-preview"><span className="sound-eyebrow">Your mix</span><p>{generatedPrompt}</p></div>
      </div>
      <div role="tabpanel" id={`${id}-prompt-panel`} aria-labelledby={`${id}-prompt-tab`} tabIndex={0} className="sound-mode-panel" hidden={mode !== "prompt"}>
          <div className="sound-writing">
            <div className="sound-writing-label"><label htmlFor={`${id}-prompt`}>Set the scene in your own words.</label><button type="button" onClick={() => setSoundDraft((draft) => ({ ...draft, prompt: generatedPrompt }))}>Start from this mix ↗</button></div>
            <textarea id={`${id}-prompt`} value={prompt} rows={4} maxLength={MAX_CUSTOM_PROMPT_CHARS} placeholder="A rainy evening, soft piano, warm tape, nowhere to be…" aria-describedby={`${id}-prompt-hint ${id}-prompt-count`} onChange={(event) => setSoundDraft((draft) => ({ ...draft, prompt: event.target.value.slice(0, MAX_CUSTOM_PROMPT_CHARS) }))} onKeyDown={(event) => { if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) { event.preventDefault(); applySound() } }} />
            <div className="sound-prompt-meta"><p id={`${id}-prompt-hint`}>A place, an instrument, a feeling. Keep it simple.</p><span id={`${id}-prompt-count`} aria-label={`${prompt.length} of ${MAX_CUSTOM_PROMPT_CHARS} characters`}>{prompt.length}/{MAX_CUSTOM_PROMPT_CHARS}</span></div>
            <div className="sound-ideas"><span>Try a little inspiration</span><div>{PROMPT_IDEAS.map((idea) => <button key={idea.label} type="button" onClick={() => setSoundDraft((draft) => ({ ...draft, prompt: idea.prompt }))}>{idea.label}<span aria-hidden="true">↗</span></button>)}</div></div>
          </div>
      </div>

      <div className="sound-apply-row">
        <p role="status"><span className={`sound-state-dot ${isCurrent ? "is-set" : ""}`} aria-hidden="true" />{status}</p>
        <button type="button" className="sound-apply" onClick={applySound} disabled={!canApply}>{isCurrent ? "Sound set" : mode === "builder" ? "Apply sound" : "Use prompt"}<DotGlyph name={isCurrent ? "check" : "plus"} dot={1} /></button>
      </div>

      <details className="sound-tuning">
        <summary><span><DotGlyph name="settings" dot={1} />Fine-tune the flow</span><DotGlyph name="chevron" dot={1} /></summary>
        <div className="sound-tuning-dials">
          <DotSlider label="Style match" readout={`${Math.round(controls.adherence * 100)}%`} value={Math.round(controls.adherence * 100)} onChange={(value) => setControls({ ...controls, adherence: value / 100 })} />
          <DotSlider label="Variation" readout={`${Math.round(controls.variation * 100)}%`} value={Math.round(controls.variation * 100)} onChange={(value) => setControls({ ...controls, variation: value / 100 })} />
          <p>Stay close to your sound, or give it room to wander. These dials update as you move them.</p>
        </div>
      </details>
    </section>
  )
}
