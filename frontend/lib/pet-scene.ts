// the desk companion: a small round cat, drawn as metaballs on a dot matrix.
//
// it used to be a whole room - window, moon, shelf, mug, notebook - with a cat
// squeezed into a third of the width, and at that size none of it read as
// anything. this is the opposite bet: one creature, no set, big in frame, and
// every dot spent on the shape of an animal.
//
// it is built out of blobs rather than out of pixel spans because everything
// here has to move. a span table can blink and it can shift a rank; it cannot
// squash on a downbeat, lean towards a cursor, or grow a tail that flows. a
// field of metaballs does all three for free - the parts merge where they
// overlap, so a head and a body become one soft peanut with a waist, and an
// ear grows out of a skull instead of sitting on top of it like a hat.
//
// coordinates are dots, y down, and the pose is measured up from the floor the
// cat sits on, so a hop or a squash moves everything that should move and
// nothing that should not. the numbers were tuned by looking at the output; if
// you change one, look again rather than reasoning about it.

import { BlobSet, limb, shadeSolid } from "./dot-field"

export const PET_W = 39
export const PET_H = 35

// the floor the cat sits on. every height below is measured up from here, so a
// hop moves the whole animal and a squash presses it into the ground instead
// of sliding it about.
const GROUND = 32
const MID = 19
// squash and stretch pivot on the chest rather than on the floor. pivoting on
// the floor sounds right and is not: the ear tips are twenty dots up, so a one
// tenth stretch throws them clean off the top of the panel while the belly
// barely moves.
const PIVOT = 6.0

// palette ramp, least to most lit, resolved to --dot-0..6 at paint time. the
// two themes invert lightness but not this order, so HOT is the strongest mark
// in both and OFF is always the unlit panel.
export const OFF = 0
export const FAINT = 1
export const DIM = 2
export const MID_SHADE = 3
export const LIT = 4
export const HOT = 5
export const ACCENT = 6

export type PetMood = "idle" | "focus" | "happy" | "cheer" | "sleep"

export interface PetFrame {
  mood: PetMood
  // eyes shut this frame
  blink: boolean
  // 0..1, how hard the music is hitting right now
  bob: number
  // 0..1, celebration hop - lifts the whole cat off the floor
  hop: number
  // 0..1, ear flick when a task lands
  twitch: number
  // continuously rising phase in seconds, for the tail, breath and floaters
  phase: number
  // draw notes rising off the cat
  notes: boolean
  // 0..1, celebration sparkles fading out
  sparkle: number
  // where the cat is looking, -1..1 across its field of view. this is the
  // cursor, most of the time, and it leans as well as looks.
  gazeX: number
  gazeY: number
  // 0..1, the cursor is on the cat and it has noticed
  affection: number
}

export const IDLE_FRAME: PetFrame = {
  mood: "idle",
  blink: false,
  bob: 0,
  hop: 0,
  twitch: 0,
  phase: 0,
  notes: false,
  sparkle: 0,
  gazeX: 0,
  gazeY: 0,
  affection: 0,
}

type Grid = Uint8Array

const at = (g: Grid, r: number, c: number, v: number) => {
  if (r >= 0 && r < PET_H && c >= 0 && c < PET_W) g[r * PET_W + c] = v
}

// paint only where the cat already is, for a marking that follows the body it
// is on rather than hanging in the air beside it
const over = (g: Grid, r: number, c: number, v: number) => {
  if (r >= 0 && r < PET_H && c >= 0 && c < PET_W && g[r * PET_W + c] !== OFF) {
    g[r * PET_W + c] = v
  }
}

// paint only on the fur inside the outline. a marking placed by measurement -
// the inner ear, the blush - lands on the rim as soon as the part it belongs
// to swings, and a rim dot replaced by a marking is a notch bitten out of the
// silhouette. this is the same idea as `over` with the outline protected too.
const onFur = (g: Grid, r: number, c: number, v: number) => {
  if (r >= 0 && r < PET_H && c >= 0 && c < PET_W && g[r * PET_W + c] === LIT) {
    g[r * PET_W + c] = v
  }
}

const row = (g: Grid, r: number, c0: number, c1: number, v: number) => {
  for (let c = c0; c <= c1; c++) at(g, r, c, v)
}

const overRow = (g: Grid, r: number, c0: number, c1: number, v: number) => {
  for (let c = c0; c <= c1; c++) over(g, r, c, v)
}

const box = (g: Grid, r0: number, c0: number, r1: number, c1: number, v: number) => {
  for (let r = r0; r <= r1; r++) row(g, r, c0, c1, v)
}

// an eighth note, 3 wide and 4 tall, anchored at its top-left
const note = (g: Grid, r: number, c: number, v: number) => {
  at(g, r, c + 1, v)
  at(g, r, c + 2, v)
  at(g, r + 1, c + 1, v)
  at(g, r + 2, c + 1, v)
  at(g, r + 3, c, v)
  at(g, r + 3, c + 1, v)
}

const heart = (g: Grid, r: number, c: number, v: number) => {
  at(g, r, c - 1, v)
  at(g, r, c + 1, v)
  row(g, r + 1, c - 2, c + 2, v)
  row(g, r + 2, c - 1, c + 1, v)
  at(g, r + 3, c, v)
}

const zed = (g: Grid, r: number, c: number, v: number) => {
  row(g, r, c, c + 2, v)
  at(g, r + 1, c + 1, v)
  row(g, r + 2, c, c + 2, v)
}

const star = (g: Grid, r: number, c: number, v: number) => {
  at(g, r, c, v)
  at(g, r - 1, c, v)
  at(g, r + 1, c, v)
  at(g, r, c - 1, v)
  at(g, r, c + 1, v)
}

const SPARKLE_SPOTS: Array<[number, number]> = [
  [7, 4],
  [3, 10],
  [5, 29],
  [10, 34],
  [2, 19],
]

// one grid and one field, re-struck every frame. the caller paints from the
// grid immediately and never keeps it, so handing out the same buffers saves
// two allocations a frame - a small thing thirty times a second is not a small
// thing.
const SCRATCH: Grid = new Uint8Array(PET_W * PET_H)
const FIELD = new Float32Array(PET_W * PET_H)
const BLOBS = new BlobSet()

// where the head ended up this frame, so the face can be struck onto it
let headX = MID
let headY = 14
let headR = 5.2
let faceWide = 1

export function drawPet(f: PetFrame): Grid {
  const g = SCRATCH

  poseBody(f)
  BLOBS.scatter(FIELD, PET_W, PET_H)
  // a solid cat with one dot of rim all round it and a breath of glow outside.
  // the rim is what keeps the silhouette legible at this pitch: a metaball
  // shaded by field value alone is a fog, and a fog is not an animal.
  //
  shadeSolid(FIELD, g, PET_W, PET_H, {
    fill: LIT,
    rim: HOT,
    halo: FAINT,
    haloAt: 0.4,
  })

  drawMarkings(g)
  drawFace(g, f)
  drawFloaters(g, f)

  return g
}

// ---------------------------------------------------------------------------
// the pose: every blob the cat is made of, placed for this frame
// ---------------------------------------------------------------------------
function poseBody(f: PetFrame) {
  BLOBS.reset()

  // squash and stretch. a beat presses the cat down and out; a hop pulls it in
  // and up. blobs are round, so the widening has to come from where they sit
  // rather than from scaling any one of them - spreading a pair sideways
  // widens a silhouette in a way a single circle cannot.
  //
  // only the squash shortens the cat. letting a hop lengthen it as well is
  // correct animation and wrong here: it stacks with the lift and puts the ear
  // tips through the top of the panel.
  const springy = f.hop * 0.9 - f.bob * 0.7
  const wide = 1 - springy * (springy > 0 ? 0.2 : 0.42)
  const tall = 1 + (springy < 0 ? springy * 0.16 : 0)
  const rScale = 1 + springy * 0.05
  const lift = f.hop * 2.2 - f.bob * 0.9
  const breath = Math.sin(f.phase * (f.mood === "sleep" ? 0.7 : 1.15)) * 0.2

  faceWide = wide
  const floor = GROUND - lift
  const up = (d: number) => floor - (PIVOT + (d - PIVOT) * tall)
  const out = (d: number) => MID + d * wide

  // the lean. this is most of what makes the cat feel like it is watching you:
  // pupils sliding about inside a head that never moves read as a glitch, a
  // whole head carried two dots towards the cursor reads as attention.
  const lean = f.gazeX * 2.3
  const nod = f.gazeY * 1.4

  // ---- feet: two bumps with a notch carved between them. the notch is a
  // negative blob, because two feet fused into one paddle is the single most
  // animal-destroying thing this shape can do.
  BLOBS.add(out(-3.4) + lean * 0.2, up(1.0), 1.5 * rScale)
  BLOBS.add(out(3.4) + lean * 0.2, up(1.0), 1.5 * rScale)
  BLOBS.add(MID + lean * 0.2, up(-0.4), 1.8, -0.9)

  // ---- tail, curling up off the right hip: a chain of shrinking blobs, so it
  // tapers to a point and flows instead of hinging. it swings faster and wider
  // the harder the music is going, which is the reaction you can read from
  // across the room.
  const swish = Math.sin(f.phase * (1.4 + f.bob * 2.4)) * (1 + f.bob * 1.1)
  limb(
    BLOBS,
    out(5.2),
    up(3.4),
    out(10.6) + swish * 0.4,
    up(4.4),
    out(9.8) + swish,
    up(13.4 + f.bob * 1.3),
    2.0 * rScale,
    0.5,
    8,
  )

  // ---- seat and chest. the hips are a second, wider pair below the chest, so
  // the cat is a pear rather than a ball: broad where it meets the floor,
  // narrow where the head comes out of it.
  BLOBS.add(out(-4.0) + lean * 0.3, up(4.0), 3.4 * rScale)
  BLOBS.add(out(4.0) + lean * 0.3, up(4.0), 3.4 * rScale)
  BLOBS.add(MID + lean * 0.35, up(6.4), (5.0 + breath) * rScale)

  // ---- head. three blobs in a row, and the outer two set wide: a lone circle
  // cannot be widened on a beat, and a head broader than the body it sits on
  // is most of why a drawn animal reads as young rather than as scaled down.
  headR = 5.2 * rScale
  headX = MID + lean
  headY = up(18.0) + nod
  const cheeks = 3.0 * wide + (f.mood === "happy" || f.mood === "cheer" ? 0.5 : 0)
  BLOBS.add(headX - cheeks, headY, headR)
  BLOBS.add(headX, headY, headR * 1.02)
  BLOBS.add(headX + cheeks, headY, headR)

  // ---- the neck, carved rather than drawn. head and chest are within a dot
  // of the same width, so left alone the field bridges them into one rounded
  // slab - a bag with ears on it. two negative blobs biting in from either
  // side at the join put the waist back, and a waist is most of what makes a
  // silhouette read as a creature with a head rather than as a shape.
  // it is set low and kept small on purpose. a wider bite reaches up under the
  // cheeks and takes a notch out of the jaw, and it only widens on a squash -
  // a hop narrows the cat, and a pinch that narrowed with it would close on
  // the neck like a pair of scissors.
  const pinch = 6.6 * Math.max(1, wide)
  BLOBS.add(headX - pinch, headY + 7.5, 2.0, -1.4)
  BLOBS.add(headX + pinch, headY + 7.5, 2.0, -1.4)

  // ---- ears. how far they are pricked is more of the cat's mood at a glance
  // than the face is: up when pleased, folded out flat when asleep, and the
  // near one flicking when a task lands.
  const perk =
    f.mood === "cheer" ? 1 : f.mood === "happy" ? 0.7 : f.mood === "sleep" ? -1 : f.affection * 0.6
  ear(-1, perk, wide, tall, rScale, 0)
  ear(1, perk, wide, tall, rScale, f.twitch > 0.5 ? 1 : 0)
}

// one ear, grown out of the skull as a tapering chain. side is -1 or 1.
function ear(
  side: number,
  perk: number,
  wide: number,
  tall: number,
  rScale: number,
  flick: number,
) {
  // drooping swings the tip out and down; perking stands it up and draws it in
  const spread = 2.7 - perk * 0.6 + flick * 0.9
  const height = (7.5 + perk * 0.9 - flick * 1.5) * tall
  const baseX = headX + side * 4.2 * wide
  const baseY = headY - headR * 0.55
  limb(
    BLOBS,
    baseX,
    baseY,
    // the control point bows the outer edge, which is the difference between a
    // cat's ear and a traffic cone
    baseX + side * spread * 0.3,
    baseY - height * 0.6,
    baseX + side * spread * wide,
    baseY - height,
    2.6 * rScale,
    0.5,
    5,
  )
}

// ---------------------------------------------------------------------------
// markings struck onto the silhouette once it is shaded
// ---------------------------------------------------------------------------
function drawMarkings(g: Grid) {
  const hr = Math.round(headY)
  const hc = Math.round(headX)

  // inner ears, a shade back from the fur round them. `over` keeps them on the
  // ear even as it swings, which is the whole reason they are struck late.
  for (const side of [-1, 1]) {
    const ec = Math.round(hc + side * 5.4 * faceWide)
    const er = Math.round(headY - headR * 0.55 - 3.4)
    onFur(g, er, ec, MID_SHADE)
    onFur(g, er + 1, ec, MID_SHADE)
  }

  // the collar. it sits exactly where the head and the chest join, so as well
  // as being the one spot of colour on the cat it draws a hard line across the
  // waist the field only softens.
  const collar = hr + 7
  overRow(g, collar, hc - 7, hc + 7, ACCENT)
  over(g, collar + 1, hc, HOT)

  // a pale bib down the chest, the light side of a two-tone cat. three dots
  // and a tip: any more of it and it stops being a marking and starts being a
  // hole in the animal.
  onFur(g, collar + 2, hc - 1, HOT)
  onFur(g, collar + 2, hc, HOT)
  onFur(g, collar + 2, hc + 1, HOT)
  onFur(g, collar + 3, hc, HOT)
}

function drawFace(g: Grid, f: PetFrame) {
  const happy = f.mood === "happy" || f.mood === "cheer" || f.affection > 0.55
  const hc = Math.round(headX)
  const hr = Math.round(headY)

  for (const side of [-1, 1]) {
    // the socket runs from two to five dots out, which is as much as two eyes
    // and a nose bridge fit across a head this size
    const inner = hc + side * 2
    const outer = hc + side * 5
    const ey = hr - 1

    if (f.blink || f.mood === "sleep") {
      // shut: a struck line with a tick turned down at the outer end, which is
      // a cat asleep rather than a cat with its eyes merely switched off
      const near = inner + side
      row(g, ey, Math.min(near, outer), Math.max(near, outer), OFF)
      at(g, ey + 1, outer, OFF)
    } else if (happy) {
      // the delighted arc. two dots along the top and one dropped at each end
      // is the most a four-wide socket can say, and it says it clearly - a
      // single diagonal in the same space reads as a scowl.
      at(g, ey, inner, OFF)
      at(g, ey - 1, inner + side, OFF)
      at(g, ey - 1, outer - side, OFF)
      at(g, ey, outer, OFF)
    } else {
      // wide open. the socket is a hole cut clean out of the fur and the pupil
      // is the brightest mark on the panel, so the gaze reads the same on a
      // black LED panel and on warm paper - whichever way the theme runs, the
      // thing that moves is the thing you look at.
      const narrow = f.mood === "focus" ? 1 : 0
      box(g, hr - 3 + narrow, Math.min(inner, outer), hr, Math.max(inner, outer), OFF)
      // the pupil is two dots wide inside a socket four across, so it has a
      // dot of travel each way - enough to read as a look when the head is
      // already leaning that way, and never enough to fall out of the eye
      const px = Math.round(f.gazeX * 1.1)
      const py = Math.round(f.gazeY * 0.9) + narrow
      const left = Math.min(inner, outer) + 1 + px
      box(g, hr - 2 + py, left, hr - 1 + py, left + 1, HOT)
      // the glint, in whichever corner of the socket the pupil has left empty
      if (!narrow) at(g, hr - 3, px > 0 ? Math.min(inner, outer) : Math.max(inner, outer), HOT)
    }
  }

  // ---- muzzle: a pale patch with the nose at the top of it. one dot of
  // colour on an otherwise monochrome face is worth more than any amount of
  // extra shading, and it is the dot that says which way the cat is facing.
  const my = hr + 2
  row(g, my, hc - 1, hc + 1, HOT)
  row(g, my + 1, hc - 2, hc + 2, HOT)
  at(g, my, hc, ACCENT)
  // the mouth, a small w under the nose
  at(g, my + 1, hc - 1, OFF)
  at(g, my + 1, hc + 1, OFF)
  at(g, my + 2, hc, OFF)

  // whiskers, springing off the muzzle and clearing the cheek by a dot
  for (const side of [-1, 1]) {
    const far = hc + side * 11
    const near = hc + side * 9
    row(g, my, Math.min(far, near), Math.max(far, near), DIM)
    row(g, my + 2, Math.min(far + side, near + side), Math.max(far + side, near + side), DIM)
  }

  // the flush of being noticed, warm on the cheeks under the eyes
  if (f.affection > 0.4 || happy) {
    for (const side of [-1, 1]) {
      onFur(g, hr + 1, hc + side * 5, MID_SHADE)
      onFur(g, hr + 1, hc + side * 6, MID_SHADE)
      onFur(g, hr + 2, hc + side * 5, MID_SHADE)
    }
  }
}

// ---- what the cat is feeling, floating in the space around it ----
function drawFloaters(g: Grid, f: PetFrame) {
  const hr = Math.round(headY)
  const hc = Math.round(headX)

  if (f.sparkle > 0) {
    const shown = Math.ceil(f.sparkle * SPARKLE_SPOTS.length)
    for (let i = 0; i < shown; i++) {
      const [r, c] = SPARKLE_SPOTS[i]
      if (Math.sin(f.phase * 6 + i * 1.9) > -0.5) star(g, r, c, HOT)
    }
  }

  if (f.notes && f.mood !== "sleep") {
    for (let i = 0; i < 2; i++) {
      const rise = (f.phase * 0.7 + i * 0.5) % 1
      const r = Math.round(hr - 3 - rise * 8)
      const c = hc + 11 + i * 4 + Math.round(Math.sin(rise * 4 + i) * 1.5)
      note(g, r, c, rise > 0.8 ? DIM : LIT)
    }
  }

  if (f.affection > 0) {
    for (let i = 0; i < 2; i++) {
      const rise = (f.phase * 1.1 + i * 0.5) % 1
      if (rise > f.affection) continue
      heart(g, Math.round(hr - 6 - rise * 5), hc - 12 - i * 3, rise > 0.7 ? DIM : ACCENT)
    }
  }

  if (f.mood === "sleep") {
    for (let i = 0; i < 3; i++) {
      const t = (f.phase * 0.5 + i * 0.33) % 1
      zed(g, Math.round(hr - 5 - t * 4) - i * 2, hc + 10 + i * 3, t > 0.75 ? DIM : LIT)
    }
  }
}
