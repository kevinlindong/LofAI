import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { createRequire } from "node:module"
import vm from "node:vm"

const require = createRequire(import.meta.url)
const ts = require("typescript")
const modules = new Map()
function loadModule(name) {
  if (modules.has(name)) return modules.get(name)
  const source = readFileSync(new URL(`../lib/${name.replace("./", "")}.ts`, import.meta.url), "utf8")
  const compiled = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
  }).outputText
  const context = { exports: {}, require: loadModule }
  vm.runInNewContext(compiled, context)
  modules.set(name, context.exports)
  return context.exports
}

const { DEFAULT_LISTENER_CONTROLS, MAX_CUSTOM_PROMPT_CHARS } = loadModule("./mrt-stream")
const { buildSoundPrompt, soundDraftFor, sameRecipe, describeSound, STATION_PRESETS, INSTRUMENTS, VIBES, MOODS, EFFECTS, MAX_INSTRUMENTS, MAX_EFFECTS } = loadModule("./sound-recipe")

// The largest possible mix must reach the backend without losing its final
// effect to the transport's character cap, including after adding options.
const longest = (options) => [...options].sort((a, b) => b.prompt.length - a.prompt.length)
const fullest = {
  instruments: longest(INSTRUMENTS).slice(0, MAX_INSTRUMENTS).map((option) => option.id),
  effects: longest(EFFECTS).slice(0, MAX_EFFECTS).map((option) => option.id),
  vibe: longest(VIBES)[0].id,
  mood: longest(MOODS)[0].id,
}
const longestPrompt = buildSoundPrompt(fullest)
assert.ok(longestPrompt.length <= MAX_CUSTOM_PROMPT_CHARS, `${longestPrompt.length} characters exceeds the backend cap`)
assert.ok(longestPrompt.endsWith(longest(EFFECTS)[MAX_EFFECTS - 1].prompt))
console.log("PASS every available combination fits the backend prompt cap without dropping effects")

for (const station of STATION_PRESETS) {
  const draft = soundDraftFor({ ...DEFAULT_LISTENER_CONTROLS, station: station.id })
  assert.equal(draft.mode, "builder")
  assert.ok(sameRecipe(draft.recipe, station.recipe))
}

const saved = JSON.parse(JSON.stringify({ ...DEFAULT_LISTENER_CONTROLS, station: "custom", recipe: fullest, customPrompt: longestPrompt }))
assert.equal(soundDraftFor(saved).mode, "builder")
assert.ok(sameRecipe(soundDraftFor(saved).recipe, fullest))
assert.equal(describeSound(saved).label, "Your custom mix")
const legacy = { ...DEFAULT_LISTENER_CONTROLS, station: "custom", customPrompt: "night bus, warm piano" }
assert.equal(soundDraftFor(legacy).mode, "prompt")
assert.equal(soundDraftFor(legacy).prompt, legacy.customPrompt)
assert.equal(describeSound(legacy).label, "Your own prompt")
console.log("PASS presets, saved recipes, and older manual mixes restore the correct editor mode")

for (const recipe of [null, {}, { ...fullest, instruments: ["unknown"] }, { ...fullest, effects: ["rain", "rain"] }, { ...fullest, mood: "unknown" }]) {
  const draft = soundDraftFor({ ...saved, recipe })
  assert.equal(draft.mode, "prompt")
  assert.equal(draft.prompt, longestPrompt)
}
assert.equal(soundDraftFor({ ...saved, customPrompt: "a different sound" }).mode, "prompt")
console.log("PASS invalid or stale saved recipe metadata preserves the actual prompt")
