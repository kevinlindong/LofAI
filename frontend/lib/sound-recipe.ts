import { MAX_CUSTOM_PROMPT_CHARS, type ListenerControls } from "./mrt-stream"

export const CUSTOM_STATION = "custom"
export const MAX_INSTRUMENTS = 3
export const MAX_EFFECTS = 2

export interface SoundOption {
  id: string
  label: string
  prompt: string
}

export const INSTRUMENTS: readonly SoundOption[] = [
  { id: "piano", label: "Felt piano", prompt: "felt piano" },
  { id: "guitar", label: "Jazz guitar", prompt: "jazz guitar" },
  { id: "rhodes", label: "Rhodes keys", prompt: "Rhodes" },
  { id: "saxophone", label: "Saxophone", prompt: "saxophone" },
  { id: "trumpet", label: "Muted trumpet", prompt: "muted trumpet" },
  { id: "bass", label: "Upright bass", prompt: "upright bass" },
  { id: "pads", label: "Synth pads", prompt: "synth pads" },
  { id: "music-box", label: "Music box", prompt: "music box" },
]

export const VIBES: readonly SoundOption[] = [
  { id: "lofi", label: "Lo-fi beats", prompt: "lo-fi hip hop" },
  { id: "ambient", label: "Ambient", prompt: "ambient lo-fi" },
  { id: "jazzhop", label: "Jazzhop", prompt: "lo-fi jazzhop" },
  { id: "soulful", label: "Soulful", prompt: "soulful lo-fi" },
  { id: "bossa", label: "Bossa nova", prompt: "lo-fi bossa nova" },
  { id: "dreamy", label: "Dreamy", prompt: "dreamy lo-fi" },
]

export const MOODS: readonly SoundOption[] = [
  { id: "mellow", label: "Mellow", prompt: "mellow" },
  { id: "focused", label: "Focused", prompt: "steady focused" },
  { id: "cozy", label: "Cozy", prompt: "warm intimate" },
  { id: "melancholy", label: "Melancholy", prompt: "melancholy" },
  { id: "uplifting", label: "Uplifting", prompt: "bright upbeat" },
  { id: "sleepy", label: "Sleepy", prompt: "slow gentle" },
]

export const EFFECTS: readonly SoundOption[] = [
  { id: "vinyl", label: "Vinyl crackle", prompt: "vinyl crackle" },
  { id: "tape", label: "Tape warmth", prompt: "warm tape" },
  { id: "reverb", label: "Reverb", prompt: "reverb" },
  { id: "delay", label: "Soft delay", prompt: "soft delay" },
  { id: "chorus", label: "Chorus", prompt: "chorus" },
  { id: "rain", label: "Rain", prompt: "soft rain" },
]

export interface SoundRecipe {
  instruments: string[]
  vibe: string
  mood: string
  effects: string[]
}

// Recipe metadata belongs to the interface and saved mixes. MrtStream's
// normalization sends only the existing listener controls to the backend.
export interface RadioControls extends ListenerControls {
  recipe?: SoundRecipe
}

export interface SoundDraft {
  mode: "builder" | "prompt"
  recipe: SoundRecipe
  prompt: string
}

export const STATION_PRESETS = [
  {
    id: "dusty-beats", label: "Dusty beats", description: "Warm guitar loops and an easy pocket",
    recipe: { instruments: ["guitar"], vibe: "lofi", mood: "mellow", effects: [] },
  },
  {
    id: "rainy-piano", label: "Rainy piano", description: "Spacious felt piano for quiet focus",
    recipe: { instruments: ["piano"], vibe: "ambient", mood: "melancholy", effects: [] },
  },
  {
    id: "jazz-cafe", label: "Jazz cafe", description: "Warm jazz guitar with a loose brushed swing",
    recipe: { instruments: ["guitar", "bass"], vibe: "jazzhop", mood: "cozy", effects: [] },
  },
  {
    id: "sunlit-groove", label: "Sunlit groove", description: "Muted brass and a brighter, animated beat",
    recipe: { instruments: ["trumpet", "rhodes"], vibe: "soulful", mood: "uplifting", effects: [] },
  },
] satisfies { id: string; label: string; description: string; recipe: SoundRecipe }[]

const optionPrompt = (options: readonly SoundOption[], id: string) =>
  options.find((option) => option.id === id)?.prompt ?? ""

export function buildSoundPrompt(recipe: SoundRecipe): string {
  return [
    optionPrompt(VIBES, recipe.vibe),
    optionPrompt(MOODS, recipe.mood),
    ...recipe.instruments.map((id) => optionPrompt(INSTRUMENTS, id)),
    ...recipe.effects.map((id) => optionPrompt(EFFECTS, id)),
  ].filter(Boolean).join(", ")
}

function isRecipe(value: unknown): value is SoundRecipe {
  if (!value || typeof value !== "object") return false
  const recipe = value as SoundRecipe
  const validChoices = (choices: unknown, options: readonly SoundOption[], min: number, max: number) =>
    Array.isArray(choices) && choices.length >= min && choices.length <= max &&
    new Set(choices).size === choices.length && choices.every((id) => options.some((option) => option.id === id))
  return validChoices(recipe.instruments, INSTRUMENTS, 1, MAX_INSTRUMENTS) &&
    validChoices(recipe.effects, EFFECTS, 0, MAX_EFFECTS) &&
    VIBES.some((option) => option.id === recipe.vibe) && MOODS.some((option) => option.id === recipe.mood)
}

export function soundDraftFor(controls: RadioControls): SoundDraft {
  const station = STATION_PRESETS.find((entry) => entry.id === controls.station)
  const recipe = controls.station === CUSTOM_STATION && isRecipe(controls.recipe) &&
    buildSoundPrompt(controls.recipe) === controls.customPrompt ? controls.recipe : undefined
  return {
    mode: controls.station === CUSTOM_STATION && !recipe ? "prompt" : "builder",
    recipe: recipe ?? station?.recipe ?? STATION_PRESETS[0].recipe,
    prompt: controls.station === CUSTOM_STATION && !recipe ? controls.customPrompt.slice(0, MAX_CUSTOM_PROMPT_CHARS) : "",
  }
}

export function sameRecipe(a: SoundRecipe, b: SoundRecipe): boolean {
  return a.vibe === b.vibe && a.mood === b.mood &&
    a.instruments.length === b.instruments.length && a.instruments.every((id) => b.instruments.includes(id)) &&
    a.effects.length === b.effects.length && a.effects.every((id) => b.effects.includes(id))
}

export function describeSound(controls: RadioControls) {
  return STATION_PRESETS.find((station) => station.id === controls.station) ?? {
    id: CUSTOM_STATION,
    label: controls.recipe ? "Your custom mix" : "Your own prompt",
    description: controls.customPrompt || "A soundtrack in your own words",
  }
}
