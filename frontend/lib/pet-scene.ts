// the desk companion: a cat sitting in a loaf, drawn as metaballs on a dot
// matrix.
//
// a loaf is the pose to draw at this size. a standing cat needs legs, a chest,
// a neck and a back, and at forty dots across every one of those is two dots
// wide and reads as a smudge. a loaf is one wide rounded mound with a head on
// top of it, ears, a face, two front paws tucked under the front and a tail
// laid round the side - six things, all of them big, none of them ambiguous.
// it is also the pose that squashes best: a loaf pressed down on a beat is
// still a loaf, where a sitting cat pressed down is a puddle.
//
// it is built out of blobs rather than out of pixel spans because everything
// here has to move. a span table can blink and it can shift a rank; it cannot
// settle on a downbeat, lean towards a cursor, or lay a tail that flows. a
// field of metaballs does all three for free - the parts merge where they
// overlap, so the head does not sit on the body so much as rise out of it,
// which is exactly what a loaf looks like.
//
// coordinates are dots, y down, and the pose is measured up from the surface
// the loaf is sitting on, so a hop or a squash moves everything that should
// move and nothing that should not. the numbers were tuned by looking at the
// output; if you change one, look again rather than reasoning about it.

import { BlobSet, limb, shadeSolid } from "./dot-field"

export const PET_W = 39
export const PET_H = 30

// the surface the loaf is sitting on. the field is cut off flat along it, so
// the bottom of the loaf is a straight line rather than the bottom of a
// circle - a cat in a loaf has no daylight under it, and the flat base is half
// of what makes the pose read.
const GROUND = 26.5
const MID = 19
// squash and stretch pivot on the middle of the loaf rather than on the
// ground. pivoting on the ground sounds right and is not: the ear tips are
// nineteen dots up, so a one-tenth squash moves them two dots and the belly a
// tenth of one.
const PIVOT = 5.4

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

// what the cat is doing, which is the whole of its expression. these are named
// after what you would say it was doing if you looked over at it, because that
// is also what the caption under the panel says.
export type PetMood = "idle" | "bop" | "focus" | "purr" | "happy" | "cheer" | "sleep"

export interface PetFrame {
  mood: PetMood
  // eyes shut this frame
  blink: boolean
  // 0..1, how hard the music is hitting right now
  bob: number
  // 0..1, celebration hop - lifts the whole loaf off the ground
  hop: number
  // 0..1, ear flick when something lands
  twitch: number
  // continuously rising phase in seconds, for the tail, breath and floaters
  phase: number
  // the music is playing: notes, a busier tail, a head that keeps time
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
// an eye socket, the inner ear, the blush - lands on the rim as soon as the
// part it belongs to swings, and a rim dot replaced by a marking is a notch
// bitten out of the silhouette. the eyes in particular are drawn deliberately
// large for their head, so without this they would eat the sides of it.
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

const furRow = (g: Grid, r: number, c0: number, c1: number, v: number) => {
  for (let c = c0; c <= c1; c++) onFur(g, r, c, v)
}

const furBox = (g: Grid, r0: number, c0: number, r1: number, c1: number, v: number) => {
  for (let r = r0; r <= r1; r++) furRow(g, r, c0, c1, v)
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
  [5, 7],
  [2, 13],
  [4, 27],
  [8, 33],
  [1, 20],
]

// one grid and one field, re-struck every frame. the caller paints from the
// grid immediately and never keeps it, so handing out the same buffers saves
// two allocations a frame - a small thing thirty times a second is not a small
// thing.
const SCRATCH: Grid = new Uint8Array(PET_W * PET_H)
const FIELD = new Float32Array(PET_W * PET_H)
const BLOBS = new BlobSet()

// where the parts ended up this frame, so the markings can be struck onto them
let headX = MID
let headY = 15
let headR = 4.7
let faceWide = 1
let baseRow = Math.round(GROUND)
let tailFlick = 0

export function drawPet(f: PetFrame): Grid {
  const g = SCRATCH

  poseBody(f)
  BLOBS.scatter(FIELD, PET_W, PET_H)
  // cut the field off flat along the ground. a loaf has a flat bottom, and a
  // metaball never will on its own.
  if (baseRow + 1 < PET_H) FIELD.fill(0, (baseRow + 1) * PET_W)

  // a solid cat with one dot of rim all round it and a breath of glow outside.
  // the rim is what keeps the silhouette legible at this pitch: a metaball
  // shaded by field value alone is a fog, and a fog is not an animal.
  shadeSolid(FIELD, g, PET_W, PET_H, {
    fill: LIT,
    rim: HOT,
    halo: FAINT,
    haloAt: 0.4,
  })

  drawShadow(g, f)
  drawMarkings(g, f)
  drawFace(g, f)
  drawFloaters(g, f)

  return g
}

// ---------------------------------------------------------------------------
// the pose: every blob the loaf is made of, placed for this frame
// ---------------------------------------------------------------------------
function poseBody(f: PetFrame) {
  BLOBS.reset()

  // squash and stretch. a beat presses the loaf down and out; a hop pulls it
  // in and up. blobs are round, so the widening has to come from where they
  // sit rather than from scaling any one of them - spreading a row of them
  // sideways widens a silhouette in a way a single circle cannot.
  //
  // only the squash shortens the cat. letting a hop lengthen it as well is
  // correct animation and wrong here: it stacks with the lift and puts the ear
  // tips through the top of the panel.
  const springy = f.hop * 0.9 - f.bob * 0.7
  const wide = 1 - springy * (springy > 0 ? 0.18 : 0.34)
  const tall = 1 + (springy < 0 ? springy * 0.17 : 0)
  const rScale = 1 + springy * 0.05
  const lift = f.hop * 3.0 - f.bob * 0.6
  const breath = Math.sin(f.phase * (f.mood === "sleep" ? 0.7 : 1.15)) * 0.2
  // a cat being fussed vibrates. one dot, at a rate nothing else on the panel
  // moves at, and it reads as purring from across the room.
  const purr = f.mood === "purr" ? Math.round(Math.sin(f.phase * 22)) * 0.35 : 0
  tailFlick = Math.sin(f.phase * (1.2 + f.bob * 2.6)) * (0.7 + f.bob * 1.7)

  faceWide = wide
  const floor = GROUND - lift
  baseRow = Math.round(floor)
  const up = (d: number) => floor - (PIVOT + (d - PIVOT) * tall)
  const out = (d: number) => MID + d * wide

  // the lean. this is most of what makes the cat feel like it is watching you:
  // pupils sliding about inside a head that never moves read as a glitch, a
  // whole head carried two dots towards the cursor reads as attention.
  const lean = f.gazeX * 2.0
  const nod = f.gazeY * 1.1
  // keeping time. the head rocks side to side and settles into the loaf on the
  // beat, which is the whole of what a cat sitting in front of a speaker does.
  const sway = f.notes ? Math.sin(f.phase * 2.4) * (0.4 + f.bob * 1.6) : 0

  // ---- the tail. only the last third of it: the rest is behind the loaf,
  // which is where a cat sitting like this keeps it. it sweeps out along the
  // ground past the flank and then curls up, and the curl is the part that
  // moves - faster and further the louder the music, which is the reaction you
  // can read from across the room.
  //
  // it was drawn across the front of the body for a while, since a loaf this
  // wide leaves little panel either side of it. a band of darker fur laid over
  // a body reads as a smudge on the cat, not as a tail behind it; four dots of
  // clear panel is worth more than any amount of shading.
  limb(
    BLOBS,
    out(10.0),
    up(1.6),
    out(15.2),
    up(1.2),
    out(14.6),
    up(5.4 + tailFlick),
    1.7 * rScale,
    0.7,
    6,
  )

  // ---- the loaf, in four ranks. the ranks are the whole trick: blobs simply
  // arced from the middle down to the ends give sides that slope straight into
  // the head, and the cat comes out a pyramid with ears on it. a loaf is a box
  // with the corners taken off - nearly vertical down the flanks, flat across
  // the top - and getting that out of circles means stacking a narrow rank
  // right above a wide one rather than spreading one rank thinner.
  //
  // rank B is the one that does the work. its top is set just under the jaw on
  // purpose, so the flanks step out below the head instead of growing out of
  // it.
  const sag = lean * 0.22
  for (const d of [-8.2, -3.0, 3.0, 8.2]) BLOBS.add(out(d) + sag, up(2.4), 4.6 * rScale)
  for (const d of [-8.0, 8.0]) BLOBS.add(out(d) + sag, up(3.6), 3.8 * rScale)
  for (const d of [-6.6, 6.6]) BLOBS.add(out(d) + sag, up(5.0), 4.2 * rScale)
  for (const d of [-4.6, 4.6]) BLOBS.add(out(d) + sag, up(6.0), 4.6 * rScale)
  for (const d of [-1.5, 1.5]) BLOBS.add(out(d) + sag, up(6.3), (4.7 + breath) * rScale)

  // ---- head. it rises out of the loaf rather than sitting on it, so the
  // overlap is deliberately heavy - a visible neck here would undo the pose.
  // three blobs again, because a lone circle cannot be widened on a beat and a
  // cat's head is a shade wider than it is tall in any case.
  headR = 4.6 * rScale
  headX = MID + lean + sway + purr
  headY = up(12.6) + nod + f.bob * 1.2
  const spread = 1.9 * wide + (f.mood === "happy" || f.mood === "cheer" ? 0.4 : 0)
  BLOBS.add(headX - spread, headY, headR)
  BLOBS.add(headX, headY, headR * 1.02)
  BLOBS.add(headX + spread, headY, headR)
  // cheeks: a lower, wider pair. a cat's head is broadest at the jaw, and
  // without these the skull is an egg and the whole face reads as a fox.
  const jowl = 3.3 * wide
  BLOBS.add(headX - jowl, headY + 1.5, 3.5 * rScale)
  BLOBS.add(headX + jowl, headY + 1.5, 3.5 * rScale)

  // ---- the jaw, carved rather than drawn. blobs merging is exactly what the
  // rest of this file is built on, and here it is the enemy: a narrow head
  // above a wide loaf fuses into one smooth cone, so the outline runs in an
  // unbroken diagonal from the ear tips to the bottom corners and the whole
  // cat reads as a pyramid. two bites just outside the cheeks hold the width
  // in under the face, and the flanks then step out below them - which is the
  // one place a loaf's outline is allowed a corner.
  BLOBS.add(headX - 8.4 * wide, headY + 3.6, 2.4, -1.2)
  BLOBS.add(headX + 8.4 * wide, headY + 3.6, 2.4, -1.2)

  // ---- ears. how far they are pricked is more of the cat's mood at a glance
  // than the face is: up when pleased, folded out flat when asleep, and one of
  // them flicking whenever something happens.
  const perk =
    f.mood === "cheer"
      ? 1
      : f.mood === "happy" || f.mood === "purr"
        ? 0.7
        : f.mood === "sleep"
          ? -1
          : f.mood === "focus"
            ? 0.5
            : f.affection * 0.5
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
  const spread = 2.6 - perk * 0.7 + flick * 0.9
  const height = (7.2 + perk * 0.9 - flick * 1.5) * tall
  const baseX = headX + side * 3.4 * wide
  const baseY = headY - headR * 0.5
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
    2.3 * rScale,
    0.5,
    5,
  )
}

// the ground the loaf is sitting on, drawn as a shadow rather than as a line.
// it is here for the hop: without something staying put underneath, a cat that
// lifts two dots reads as a cat that has been nudged rather than as a cat that
// has jumped.
function drawShadow(g: Grid, f: PetFrame) {
  const r = Math.round(GROUND) + 1
  const half = Math.round(10 - f.hop * 5)
  for (let c = MID - half; c <= MID + half; c++) {
    // thinned at the ends, so it reads as a pool rather than as a plank
    at(g, r, c, Math.abs(c - MID) > half - 2 ? FAINT : DIM)
  }
}

// ---------------------------------------------------------------------------
// markings struck onto the silhouette once it is shaded
// ---------------------------------------------------------------------------
function drawMarkings(g: Grid, f: PetFrame) {
  const hr = Math.round(headY)
  const hc = Math.round(headX)

  // inner ears, a shade back from the fur round them. `onFur` keeps them on
  // the ear even as it swings, and off the outline, which is the whole reason
  // they are struck late.
  for (const side of [-1, 1]) {
    const ec = Math.round(hc + side * 4.8 * faceWide)
    const er = Math.round(headY - headR * 0.5 - 3.2)
    onFur(g, er, ec, MID_SHADE)
    onFur(g, er + 1, ec, MID_SHADE)
  }

  // the collar. with the head grown out of the loaf there is no neck to see,
  // so the collar is the only thing drawing the line between the two - as well
  // as being the one spot of colour on an otherwise amber cat.
  const collar = hr + 5
  overRow(g, collar, hc - 5, hc + 5, ACCENT)
  onFur(g, collar + 1, hc, HOT)

  // ---- the two front paws, tucked under the front of the loaf. they are the
  // detail that says loaf rather than lump: a cat sitting like this has its
  // paws folded away with just the toes out in front.
  // one clear of the base line: that bottom row is all rim, and a pale paw
  // struck onto the rim is a paw you cannot see
  const pr = baseRow - 2
  for (const side of [-1, 1]) {
    const near = MID + side * 2
    const far = MID + side * 5
    furRow(g, pr, Math.min(near, far), Math.max(near, far), HOT)
    furRow(g, pr + 1, Math.min(near, far), Math.max(near, far), HOT)
    // one toe split each. two would be truer and at four dots across it comes
    // out as a comb. struck with `at` rather than `onFur`, because by now the
    // paw it is splitting is no longer fur.
    at(g, pr, MID + side * 4, MID_SHADE)
    at(g, pr + 1, MID + side * 4, MID_SHADE)
  }

  // a pale bib on the chest, the light side of a two-tone cat, running down
  // from the collar to meet the paws
  if (f.mood !== "sleep") {
    furRow(g, collar + 2, hc - 1, hc + 1, HOT)
    onFur(g, collar + 3, hc, HOT)
  }
}

function drawFace(g: Grid, f: PetFrame) {
  const beaming = f.mood === "happy" || f.mood === "cheer" || f.mood === "purr"
  const hc = Math.round(headX)
  const hr = Math.round(headY)

  // how far the lids have come down. the eyes are the loudest thing on the
  // panel, so most of the difference between one mood and the next is here.
  const lid = f.mood === "focus" ? 2 : f.mood === "bop" ? 1 : 0

  for (const side of [-1, 1]) {
    // the socket runs from two to five dots out. that is very nearly the whole
    // width of the head, which is the point - eyes drawn to a sensible scale
    // read as an animal, and eyes drawn much too big read as a pet.
    const inner = hc + side * 2
    const outer = hc + side * 5
    const lo = Math.min(inner, outer)
    const hi = Math.max(inner, outer)
    const bottom = hr
    const top = hr - 3 + lid

    if (f.blink || f.mood === "sleep") {
      // shut: a struck line with a tick turned down at the outer end, which is
      // a cat asleep rather than a cat with its eyes merely switched off
      furRow(g, bottom - 1, lo, hi, OFF)
      onFur(g, bottom, outer, OFF)
    } else if (beaming) {
      // the delighted arc. two dots along the top and one dropped at each end
      // is the most a four-wide socket can say, and it says it clearly - a
      // single diagonal in the same space reads as a scowl.
      onFur(g, bottom, inner, OFF)
      onFur(g, bottom - 1, inner + side, OFF)
      onFur(g, bottom - 1, outer - side, OFF)
      onFur(g, bottom, outer, OFF)
    } else {
      // open. the socket is a hole cut clean out of the fur and the pupil is
      // the brightest mark on the panel, so the gaze reads the same on a black
      // LED panel and on warm paper - whichever way the theme runs, the thing
      // that moves is the thing you look at.
      furBox(g, top, lo, bottom, hi, OFF)
      // the pupil is two dots wide inside a socket four across, so it has a
      // dot of travel each way - enough to read as a look when the head is
      // already leaning that way, and never enough to fall out of the eye
      const px = Math.round(f.gazeX * 1.1)
      const py = Math.max(top - bottom + 1, Math.min(0, Math.round(f.gazeY * 0.9)))
      if (lid < 2) row(g, bottom - 2 + py, lo + 1 + px, lo + 2 + px, HOT)
      row(g, bottom - 1 + py, lo + 1 + px, lo + 2 + px, HOT)
      row(g, bottom + py, lo + 1 + px, lo + 2 + px, HOT)
      // the glint, in whichever corner of the socket the pupil has left empty
      // ...and mirrored, so the light is coming from one place rather than
      // from wherever the maths happened to put it
      if (lid < 2) at(g, top, px * side > 0 ? inner : outer, HOT)
    }
  }

  // ---- muzzle: a pale patch with the nose at the top of it. one dot of
  // colour on an otherwise monochrome face is worth more than any amount of
  // extra shading, and it is the dot that says which way the cat is facing.
  const my = hr + 1
  furRow(g, my, hc - 1, hc + 1, HOT)
  furRow(g, my + 1, hc - 2, hc + 2, HOT)
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
  if (f.affection > 0.4 || beaming) {
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
      const r = Math.round(hr - 5 - rise * 7)
      const c = hc + 9 + i * 4 + Math.round(Math.sin(rise * 4 + i) * 1.5)
      note(g, r, c, rise > 0.8 ? DIM : LIT)
    }
  }

  if (f.affection > 0) {
    for (let i = 0; i < 2; i++) {
      const rise = (f.phase * 1.1 + i * 0.5) % 1
      if (rise > f.affection) continue
      heart(g, Math.round(hr - 7 - rise * 5), hc - 10 - i * 3, rise > 0.7 ? DIM : ACCENT)
    }
  }

  if (f.mood === "sleep") {
    for (let i = 0; i < 3; i++) {
      const t = (f.phase * 0.5 + i * 0.33) % 1
      zed(g, Math.round(hr - 6 - t * 4) - i * 2, hc + 8 + i * 3, t > 0.75 ? DIM : LIT)
    }
  }
}
