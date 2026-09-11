"use client"

import Link from "next/link"
import { useEffect, useRef, useState, type CSSProperties, type FormEvent, type ReactNode } from "react"
import { AsciiAmbience } from "@/components/ascii-ambience"
import { DotGlyph } from "@/components/dot-glyph"
import { DotSlider } from "@/components/dot-slider"
import { DotVisualizer } from "@/components/dot-visualizer"
import { SoundEditor } from "@/components/sound-editor"
import { Pet } from "@/components/pet"
import { PomodoroTimer } from "@/components/pomodoro-timer"
import { TodoList } from "@/components/todo-list"
import { useRadio } from "@/components/radio-provider"
import { applyDesignAppearance, DESIGNS, DESIGN_TONES, type NewDesign, type DesignTone } from "@/lib/designs"
import { MAX_CUSTOM_PROMPT_CHARS } from "@/lib/mrt-stream"
import { CUSTOM_STATION, describeSound, STATION_PRESETS, type RadioControls } from "@/lib/sound-recipe"

interface Preferences {
  tone: DesignTone
  companion: boolean
  tasks: boolean
}

interface SavedMix {
  name: string
  controls: RadioControls
  volume: number
}

const DEFAULT_PREFERENCES: Preferences = { tone: "default", companion: true, tasks: true }
const PRESETS = [
  { name: "Deep focus", station: "rainy-piano", drums: false, volume: 55, description: "Piano, no distractions" },
  { name: "Easy flow", station: "dusty-beats", drums: true, volume: 70, description: "A little rhythm, a little room" },
  { name: "Bright ideas", station: "sunlit-groove", drums: true, volume: 75, description: "A lift for your afternoon" },
] as const

const STATION_MOODS = ["Mellow / warm", "Quiet / spacious", "Loose / soulful", "Bright / easy"]

function StationArt({ station, small = false }: { station: string; small?: boolean }) {
  return (
    <span className={`station-art art-${station}${small ? " art-small" : ""}`} aria-hidden="true">
      <i /><i /><i /><i /><i />
    </span>
  )
}

function Status({ compact = false }: { compact?: boolean }) {
  const { isLive, label } = useRadio()
  return <span className={`radio-status ${isLive ? "is-live" : ""} ${compact ? "status-compact" : ""}`} role="status"><i />{label}</span>
}

function RecordPlayer({ children }: { children?: ReactNode }) {
  const { isLive, getSpectrum, wantsAudio, togglePlayback } = useRadio()
  return (
    <div className="record-stage">
      <div className="record-orbit" aria-hidden="true" />
      <DotVisualizer getSpectrum={getSpectrum} active={isLive}>
        <div className="record-center">
          <span className="record-wordmark">lofAI</span>
          <button className="design-play" type="button" onClick={togglePlayback} aria-label={wantsAudio ? "Pause" : "Play"} aria-pressed={wantsAudio}>
            <DotGlyph name={wantsAudio ? "pause" : "play"} dot={3} />
          </button>
          <span className="record-caption">{wantsAudio ? "hold this moment" : "press play, drift away"}</span>
        </div>
      </DotVisualizer>
      {children}
    </div>
  )
}

function VariationButton() {
  const { wantsAudio, streamState, requestVariation } = useRadio()
  return (
    <button type="button" className="variation-button" onClick={requestVariation} disabled={!wantsAudio || streamState.variationPending} aria-label="Skip to a new music variation">
      <DotGlyph name="rewind" dot={1} />
      {streamState.variationPending ? "Finding a new take…" : "New take"}
      <span aria-hidden="true">↗</span>
    </button>
  )
}

function StationList({ layout = "list" }: { layout?: "list" | "tiles" | "keys" }) {
  const { controls, selectStation, isLive } = useRadio()
  return (
    <>
      <div className={`station-selector stations-${layout}`} role="group" aria-label="Choose your station">
        {STATION_PRESETS.map((station, index) => {
          const selected = controls.station === station.id
          return (
            <button key={station.id} type="button" className={`station-option ${selected ? "is-selected" : ""}`} aria-pressed={selected} onClick={() => selectStation(station.id)}>
              <span className="station-index">0{index + 1}</span>
              {layout === "list" && <StationArt station={station.id} small />}
              <span className="station-copy"><strong>{station.label}</strong><span>{STATION_MOODS[index]}</span></span>
              <span className={`station-indicator ${selected && isLive ? "is-live" : ""}`} aria-hidden="true"><i /><i /><i /></span>
            </button>
          )
        })}
      </div>
      <SoundEditor showPresets={false} />
    </>
  )
}

function Mixer() {
  const { controls, setControls, volume, setVolume, toggleMute } = useRadio()
  return (
    <div className="design-mixer">
      <button className="mute-button" type="button" onClick={toggleMute} aria-label={volume === 0 ? "Unmute" : "Mute"} aria-pressed={volume === 0} title="Mute · M">
        <DotGlyph name="music" dot={1} /><span>{volume === 0 ? "Off" : "Vol"}</span>
      </button>
      <div className="mixer-volume"><DotSlider label="Volume" readout={`${volume}%`} value={volume} onChange={setVolume} /></div>
      <button type="button" className="drum-switch" aria-pressed={controls.drums} onClick={() => setControls({ ...controls, drums: !controls.drums })}>
        <span>Drums</span><span className="switch-track"><i /></span><span>{controls.drums ? "On" : "Off"}</span>
      </button>
    </div>
  )
}

function ScenePresets() {
  const { controls, selectStation, setVolume } = useRadio()
  return (
    <div className="scene-presets" role="group" aria-label="Listening presets">
      {PRESETS.map((preset, index) => (
        <button key={preset.name} type="button" title={preset.description} aria-pressed={controls.station === preset.station && controls.drums === preset.drums} onClick={() => { selectStation(preset.station, { drums: preset.drums }); setVolume(preset.volume) }}>
          <span className="preset-mark" aria-hidden="true">{["◒", "≈", "✳"][index]}</span>{preset.name}
        </button>
      ))}
    </div>
  )
}

function SleepTimer({ expanded = false }: { expanded?: boolean }) {
  const { sleepEndsAt, sleepRemaining, setSleepTimer } = useRadio()
  const [minutes, setMinutes] = useState(30)
  return (
    <div className={`sleep-control ${expanded ? "sleep-expanded" : ""}`}>
      <DotGlyph name="moon" dot={1} />
      <span>{sleepEndsAt ? `Sleep in ${Math.floor(sleepRemaining / 60)}:${String(sleepRemaining % 60).padStart(2, "0")}` : "Sleep timer"}</span>
      {sleepEndsAt ? (
        <button type="button" onClick={() => setSleepTimer(0)} aria-label="Cancel sleep timer">Cancel</button>
      ) : (
        <>
          <select aria-label="Sleep timer duration" value={minutes} onChange={(event) => setMinutes(Number(event.target.value))}>
            <option value={15}>15 min</option><option value={30}>30 min</option><option value={60}>60 min</option><option value={90}>90 min</option>
          </select>
          <button type="button" onClick={() => setSleepTimer(minutes)}>Set</button>
        </>
      )}
    </div>
  )
}

function Widgets({ preferences, layout = "row" }: { preferences: Preferences; layout?: "row" | "column" }) {
  const { petSignal, focusMode, isLive, getLevel, handlePetEvent, setFocusMode } = useRadio()
  return (
    <div className={`design-widgets widgets-${layout}`} id="workspace">
      <section id="focus-timer" className="widget timer-card" tabIndex={-1}>
        <div className="widget-caption"><span>01 / A little structure</span><DotGlyph name="timer" dot={1} /></div>
        <PomodoroTimer onRunningChange={setFocusMode} />
      </section>
      <section id="tasks" className="widget tasks-card" hidden={!preferences.tasks} tabIndex={-1}>
        <div className="widget-caption"><span>02 / One thing at a time</span><DotGlyph name="list" dot={1} /></div>
        <TodoList onEvent={handlePetEvent} />
      </section>
      <section className="widget companion-card" hidden={!preferences.companion}>
        <div className="widget-caption"><span>03 / Good company</span><span className="companion-dot" /></div>
        <Pet signal={petSignal} focus={focusMode} playing={isLive} getLevel={getLevel} />
        <p className="companion-note">A quiet companion. Always on your side.</p>
      </section>
    </div>
  )
}

function DesignHeader({ design, openSettings, openWorkspace }: { design: NewDesign; openSettings: () => void; openWorkspace: () => void }) {
  return (
    <header className="design-header">
      <Link href={`/${design.id}`} prefetch={false} className="design-brand" aria-label={`${design.name} by lofAI`}><span>lofAI<span className="brand-dot">®</span></span><span className="brand-edition">{design.name}</span></Link>
      <nav className="design-switcher" aria-label="Choose your design">
        <span className="switcher-label">Spaces</span>
        {DESIGNS.map((entry) => <Link key={entry.id} href={`/${entry.id}`} prefetch={false} aria-current={entry.id === design.id ? "page" : undefined} title={`${entry.id} — ${entry.name}: ${entry.description}`}><span className="sr-only">{entry.name} design </span>0{entry.id}</Link>)}
      </nav>
      <div className="design-header-actions">
        <button type="button" className="workspace-link" onClick={openWorkspace}><DotGlyph name="list" dot={1} /><span>Your workspace</span></button>
        <button type="button" className="customize-button" onClick={openSettings}><DotGlyph name="settings" dot={1} /><span>Make it yours</span></button>
      </div>
    </header>
  )
}

function Sunday({ preferences }: { preferences: Preferences }) {
  const { controls } = useRadio()
  const station = describeSound(controls)
  return (
    <>
      <div className="sunday-heading">
        <div><p className="design-eyebrow">An independent listening room</p><h1>A little less rush.<br /><em>A little more rhythm.</em></h1></div>
        <p className="sunday-intro">For the coffee that went cold.<br />The book you can’t put down.<br />The good kind of getting lost.</p>
      </div>
      <div className="sunday-main" id="radio">
        <section className="sunday-record" aria-label="Music player">
          <div className="record-topline"><span>PERSONAL PRESSING / VOL. 01</span><span>∞ RPM</span></div>
          <RecordPlayer />
          <div className="sunday-record-footer"><div><span className="design-eyebrow">On the turntable</span><h2>{station.label}</h2></div><Status /></div>
          <span className="record-corner" aria-hidden="true">✳</span>
        </section>
        <section className="sunday-library" aria-label="Stations and sound">
          <div className="section-heading"><div><p className="design-eyebrow">Find your frequency</p><h2>What’s your mood?</h2></div><span className="small-index">01—04</span></div>
          <StationList />
          <div className="library-note"><span>Always unfolding. Never on repeat.</span><VariationButton /></div>
          <Mixer />
          <div className="sunday-postscript"><span aria-hidden="true">↳</span><p>Your own little corner of the internet.<br />Stay as long as you like.</p></div>
        </section>
      </div>
      <div className="workspace-heading"><h2>Make a little room.</h2><span>For your thoughts, your work, yourself.</span></div>
      <Widgets preferences={preferences} />
    </>
  )
}

function Form({ preferences }: { preferences: Preferences }) {
  const { controls } = useRadio()
  const station = describeSound(controls)
  return (
    <div className="form-shell">
      <aside className="form-sidebar">
        <p className="design-eyebrow">Less noise.<br />More possibility.</p>
        <nav aria-label="Workspace sections"><a href="#radio"><span>01</span>Listen <span>↗</span></a><a href="#focus-timer"><span>02</span>Focus <span>↗</span></a>{preferences.tasks && <a href="#tasks"><span>03</span>Make progress <span>↗</span></a>}</nav>
        <div className="form-sidebar-bottom"><div className="form-symbol" aria-hidden="true">✳</div><p>GOOD WORK<br />TAKES A LITTLE<br />SPACE.</p><span>YOUR PERSONAL SOUND SYSTEM</span></div>
      </aside>
      <div className="form-content">
        <div className="form-heading"><h1>Find your<br /><span>headspace.</span></h1><p>One intention.<br />An open afternoon.<br />A sound to settle into.</p></div>
        <div className="form-grid">
          <section className="form-listening" id="radio" aria-label="Listening workspace">
            <div className="form-section-label"><span>01 — SET THE TONE</span><span>∞</span></div>
            <ScenePresets />
            <div className="form-player"><RecordPlayer /><div className="form-now-playing"><Status /><p className="design-eyebrow">In your headphones</p><h2>{station.label}</h2><p>{station.description}.</p><VariationButton /></div></div>
            <div className="form-station-label"><span className="design-eyebrow">Or choose a station</span><span>Four ways into your flow ↙</span></div>
            <StationList layout="tiles" />
            <Mixer />
            <div className="form-bottom-note"><span className="form-dot" /><span>Made in the moment.<br />A different soundtrack, every time.</span><span aria-hidden="true">↗</span></div>
          </section>
          <Widgets preferences={preferences} layout="column" />
        </div>
      </div>
    </div>
  )
}

function Signal({ preferences }: { preferences: Preferences }) {
  const { controls, isLive } = useRadio()
  const stationIndex = STATION_PRESETS.findIndex((entry) => entry.id === controls.station)
  const station = describeSound(controls)
  return (
    <>
      <div className="signal-heading"><p className="design-eyebrow">A modern ritual. An analog state of mind.</p><h1>Good sound.<br /><span>No end in sight.</span></h1><span className="signal-seal">HIGH<br />FIDELITY<span>∞</span></span></div>
      <section className="receiver" id="radio" aria-label="Radio receiver">
        <div className="receiver-top"><span className="receiver-logo">lofAI <span>STEREO RECEIVER</span></span><span>MODEL 004 / CONTINUOUS PLAY</span><span className={`receiver-lamp ${isLive ? "is-live" : ""}`}>{isLive ? "ON AIR" : "STANDBY"}</span></div>
        <div className="tuner" aria-hidden="true"><div className="tuner-labels"><span>01 — DUSTY</span><span>02 — RAINY</span><span>03 — JAZZ</span><span>04 — SUNLIT</span></div><div className="tuner-scale"><span hidden={stationIndex < 0} style={{ "--tuner-position": `${12.5 + stationIndex * 25}%` } as CSSProperties} /></div><span className="tuner-band">PERSONAL FREQUENCY / FM ∞</span></div>
        <div className="receiver-body">
          <div className="receiver-disc"><span className="receiver-disc-label">LIQUID FREQUENCY DISPLAY</span><RecordPlayer /><span className="receiver-disc-bottom">GENERATIVE STEREO SOUND</span></div>
          <div className="receiver-controls">
            <div className="signal-display"><span className="display-label">{stationIndex < 0 ? "CUSTOM MIX" : `CHANNEL 0${stationIndex + 1}`} <span>STEREO · ∞</span></span><h2>{station.label}</h2><p>{station.description}</p><Status /></div>
            <div className="hardware-label"><span>STATION MEMORY</span><span>PUSH TO TUNE ↓</span></div>
            <StationList layout="keys" />
            <Mixer />
            <div className="receiver-bottom"><span>ONE OF ONE.<br />EVERY SINGLE TIME.</span><VariationButton /></div>
          </div>
        </div>
        <span className="receiver-screw screw-left" aria-hidden="true" /><span className="receiver-screw screw-right" aria-hidden="true" />
      </section>
      <div className="signal-rack-label"><span>AUXILIARY EQUIPMENT</span><span>YOUR DESK, TUNED IN.</span></div>
      <Widgets preferences={preferences} />
    </>
  )
}

function Customization({ design, preferences, updatePreferences, dialogRef }: {
  design: NewDesign; preferences: Preferences; updatePreferences: (next: Partial<Preferences>) => void; dialogRef: React.RefObject<HTMLDialogElement>
}) {
  const { controls, restoreMix, volume, setVolume } = useRadio()
  const [mixName, setMixName] = useState("")
  const [mixes, setMixes] = useState<SavedMix[]>([])
  const [notice, setNotice] = useState("")

  useEffect(() => {
    try {
      const saved: unknown = JSON.parse(localStorage.getItem("lofai.saved-mixes") || "[]")
      if (Array.isArray(saved)) setMixes(saved.filter((mix): mix is SavedMix =>
        typeof mix?.name === "string" && typeof mix.controls?.drums === "boolean" &&
        (STATION_PRESETS.some((s) => s.id === mix.controls?.station) ||
          (mix.controls?.station === CUSTOM_STATION && typeof mix.controls.customPrompt === "string" && mix.controls.customPrompt.length <= MAX_CUSTOM_PROMPT_CHARS)) &&
        Number.isFinite(mix.volume) && mix.volume >= 0 && mix.volume <= 100,
      ).slice(0, 6))
    } catch { /* Saved mixes are optional; controls remain usable. */ }
  }, [])

  const storeMixes = (next: SavedMix[], message: string) => {
    setMixes(next)
    try { localStorage.setItem("lofai.saved-mixes", JSON.stringify(next)); setNotice(message) }
    catch { setNotice("Available for this visit. Browser storage is unavailable.") }
  }

  const saveMix = (event: FormEvent) => {
    event.preventDefault()
    const name = mixName.trim() || describeSound(controls).label
    if (mixes.length >= 6) { setNotice("Your six mix slots are full. Remove one to save another."); return }
    storeMixes([...mixes, { name, controls: { ...controls }, volume }], `“${name}” saved.`)
    setMixName("")
  }

  return (
    <dialog ref={dialogRef} className="customization-dialog" aria-labelledby="customization-title" onClick={(event) => { if (event.target === event.currentTarget) dialogRef.current?.close() }}>
      <div className="customization-inner">
        <div className="dialog-heading"><div><p className="design-eyebrow">{design.name} / Your space</p><h2 id="customization-title">Make yourself at home.</h2></div><button type="button" className="close-button" aria-label="Close customization" onClick={() => dialogRef.current?.close()}><DotGlyph name="cross" dot={2} /></button></div>
        <p className="dialog-intro">A few small changes. A space that feels like you.</p>
        <fieldset><legend>01 / Color story</legend><div className="tone-options">{DESIGN_TONES[design.slug].map((tone) => <button key={tone.id} type="button" aria-pressed={preferences.tone === tone.id} onClick={() => updatePreferences({ tone: tone.id })}><span style={{ backgroundColor: tone.color }}>{preferences.tone === tone.id && <DotGlyph name="check" dot={2} color="#fff" />}</span>{tone.name}</button>)}</div></fieldset>
        <fieldset><legend>02 / Set the mood</legend><ScenePresets /><p className="settings-helper">Instantly sets the station, drums, and volume.</p></fieldset>
        <fieldset><legend>03 / Keep a favorite</legend><form className="save-mix-form" onSubmit={saveMix}><input aria-label="Mix name" placeholder="Give this mix a name" value={mixName} onChange={(event) => setMixName(event.target.value)} maxLength={36} /><button type="submit">Save mix <span>+</span></button></form><p className="settings-helper">Saves your applied sound, instruments, mood, effects, dials, drums, and volume.</p><div className="saved-mixes">{mixes.map((mix, index) => <div key={`${index}-${mix.name}`}><button type="button" onClick={() => { restoreMix(mix.controls); setVolume(mix.volume); setNotice(`“${mix.name}” loaded.`) }}><DotGlyph name="music" dot={1} /><span>{mix.name}</span><span>↗</span></button><button type="button" aria-label={`Remove saved mix ${mix.name}`} onClick={() => storeMixes(mixes.filter((_, i) => i !== index), `“${mix.name}” removed.`)}><DotGlyph name="cross" dot={1} /></button></div>)}</div><p className="settings-notice" role="status">{notice}</p></fieldset>
        <fieldset><legend>04 / Your workspace</legend>{([ ["tasks", "Task list", "A place for what’s on your mind."], ["companion", "A little company", "Keep your dot-matrix cat close."] ] as const).map(([key, label, description]) => <label className="preference-toggle" key={key}><span><strong>{label}</strong><small>{description}</small></span><input type="checkbox" checked={preferences[key]} onChange={(event) => updatePreferences({ [key]: event.target.checked })} /></label>)}</fieldset>
        <fieldset><legend>05 / Wind down</legend><SleepTimer expanded /><p className="settings-helper">Music pauses when the time is up.</p></fieldset>
        <div className="keyboard-hints"><span><kbd>space</kbd> play / pause</span><span><kbd>M</kbd> mute</span><span><kbd>N</kbd> new take</span></div>
      </div>
    </dialog>
  )
}

export function DesignWorkspace({ design }: { design: NewDesign }) {
  const [preferences, setPreferences] = useState<Preferences>(DEFAULT_PREFERENCES)
  const [storageNotice, setStorageNotice] = useState("")
  const dialogRef = useRef<HTMLDialogElement>(null)

  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(`lofai.design.${design.id}`) || "{}")
      if (saved && typeof saved === "object") setPreferences({
        tone: ["default", "alternate", "mono"].includes(saved.tone) ? saved.tone : "default",
        companion: typeof saved.companion === "boolean" ? saved.companion : true,
        tasks: typeof saved.tasks === "boolean" ? saved.tasks : true,
      })
    } catch { /* Use the designed defaults when storage is unavailable. */ }
  }, [design.id])

  const updatePreferences = (next: Partial<Preferences>) => {
    const updated = { ...preferences, ...next }
    setPreferences(updated)
    applyDesignAppearance(design, updated.tone)
    try { localStorage.setItem(`lofai.design.${design.id}`, JSON.stringify(updated)); setStorageNotice("") }
    catch { setStorageNotice("Your changes apply for this visit; browser storage is unavailable.") }
  }

  const openWorkspace = () => {
    document.getElementById("workspace")?.scrollIntoView({ behavior: "smooth", block: "start" })
  }

  return (
    <main className={`design-view view-${design.slug}`}>
      <a className="design-skip-link" href="#radio">Skip to music player</a>
      <AsciiAmbience />
      <div className="design-frame">
        <DesignHeader design={design} openSettings={() => dialogRef.current?.showModal()} openWorkspace={openWorkspace} />
        {design.slug === "sunday" && <Sunday preferences={preferences} />}
        {design.slug === "form" && <Form preferences={preferences} />}
        {design.slug === "signal" && <Signal preferences={preferences} />}
        <footer className="design-footer"><span>lofAI — {design.description}</span><span>Made for being here.<span className="footer-star" aria-hidden="true">✳</span></span></footer>
      </div>
      <Customization design={design} preferences={preferences} updatePreferences={updatePreferences} dialogRef={dialogRef} />
      {storageNotice && <p className="storage-notice" role="status">{storageNotice}</p>}
    </main>
  )
}
