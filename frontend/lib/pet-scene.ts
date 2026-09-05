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

import { BlobSet, limb, shadeSolid, SURFACE } from "./dot-field"

export const PET_W = 45
export const PET_H = 32

// the surface the loaf is sitting on. the field is cut off flat along it, so
// the bottom of the loaf is a straight line rather than the bottom of a
// circle - a cat in a loaf has no daylight under it, and the flat base is half
// of what makes the pose read.
const GROUND = 28
const MID = 22

// the head sits well forward of the middle of the body, and the body reaches
// away behind it to a haunch and a tail.
//
// drawn symmetrically the loaf has no front and no back: it is a mound with a
// face on it, and a mound with a face on it reads as a mound. the same
// silhouette with the head over one end has an axis, and an axis is most of
// what tells you which animal you are looking at - it is the difference
// between a shape and a creature facing left.
const HEAD_X = MID - 6
const BODY_X = MID - 0.5
// squash and stretch pivot on the middle of the loaf rather than on the
// ground. pivoting on the ground sounds right and is not: the ear tips are
// nineteen dots up, so a one-tenth squash moves them two dots and the belly a
// tenth of one.
const PIVOT = 5.0

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
  // 0..1, ear flick when something lands. continuous, not a switch: an ear
  // that snaps between two positions three times reads as a fault in the
  // panel rather than as a twitch.
  twitch: number
  // 0..1, a hand on the cat. it presses the loaf down and flattens the ears,
  // which is what a cat does when you pat it - a hop would be what a cat does
  // when you drop something.
  pat: number
  // continuously rising phase in seconds, for the tail, breath and floaters
  phase: number
  // the music is playing: notes, a busier tail, a head that keeps time
  notes: boolean
  // 0..1, how far into the music the cat is. `notes` is the switch, this is
  // the ramp behind it: the bop lifts the loaf off the ground between beats,
  // and a lift that appeared the frame the first note landed would be a jump
  // rather than a cat settling into the groove.
  groove: number
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
  pat: 0,
  phase: 0,
  notes: false,
  groove: 0,
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
  [5, 4],
  [2, 10],
  [4, 26],
  [8, 34],
  [1, 17],
]

// one grid and one field, re-struck every frame. the caller paints from the
// grid immediately and never keeps it, so handing out the same buffers saves
// two allocations a frame - a small thing thirty times a second is not a small
// thing.
const SCRATCH: Grid = new Uint8Array(PET_W * PET_H)
const FIELD = new Float32Array(PET_W * PET_H)
const BLOBS = new BlobSet()
// the head on its own, kept apart from the rest so its outline can be struck
// back over the body it is sunk into
const HEAD_BLOBS = new BlobSet()
const HEAD_FIELD = new Float32Array(PET_W * PET_H)

// where the parts ended up this frame, so the markings can be struck onto them
let headX = HEAD_X
let headY = 14
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
  drawHeadEdge(g)
  drawMarkings(g)
  drawFace(g, f)
  drawFloaters(g, f)

  return g
}

// ---------------------------------------------------------------------------
// the pose: every blob the loaf is made of, placed for this frame
// ---------------------------------------------------------------------------
function poseBody(f: PetFrame) {
  BLOBS.reset()
  HEAD_BLOBS.reset()

  // squash and stretch. a beat presses the loaf down and out; a hop pulls it
  // in and up. blobs are round, so the widening has to come from where they
  // sit rather than from scaling any one of them - spreading a row of them
  // sideways widens a silhouette in a way a single circle cannot.
  //
  // only the squash shortens the cat. letting a hop lengthen it as well is
  // correct animation and wrong here: it stacks with the lift and puts the ear
  // tips through the top of the panel.
  const springy = f.hop * 0.9 - f.bob * 0.7 - f.pat * 0.6
  // the width gets its own, much smaller, share of the beat.
  //
  // `wide` spreads the blobs out from the middle, so what it does to the
  // outline depends on how far out they already sit. the number was tuned on a
  // sitting cat whose body was four dots either side of centre, where a beat
  // moved the outline by one; the loaf that replaced it reaches twelve, where
  // the same number moves it by five. a cat that shortens by two and widens by
  // one is dipping to the music. the same cat widening by five is a pancake,
  // and that - not the shortening - is what went wrong with the bop.
  const spread = f.hop * 0.9 - f.bob * 0.12 - f.pat * 0.6
  const wide = 1 - spread * (spread > 0 ? 0.18 : 0.34)
  const tall = 1 + (springy < 0 ? springy * 0.17 : 0)
  const rScale = 1 + springy * 0.05
  // whole dots. everything that carries a marking - the head with its face on
  // it, the ground with the paws sitting on it - has to move a dot at a time,
  // because the marking can only ever be drawn a dot at a time. a head sliding
  // by a third of a dot moves its silhouette and leaves its face where it was,
  // and the result reads as a fault rather than as motion. the shape changes -
  // the squash, the ear flick, the tail - stay continuous, because nothing is
  // struck onto them.
  //
  // the beat is the exception, and it is the whole of the bop. a hop and a
  // hand are events - they land on a dot and stay there for a moment, so they
  // round. a beat is a ride: the loaf settles into the ground and comes back
  // up twice a second, and rounding that is what turned the bop into a switch
  // between a flat pose and a normal one. it stays continuous, and the
  // silhouette glides even though the markings on it still land on whole dots.
  //
  // and the bounce sits on top of it. between beats the loaf rides up off the
  // ground and each beat sets it back down, rather than the other way about:
  // the field is cut off flat along the ground, so a loaf pressed *into* it
  // loses its bottom rows to the cut instead of moving, and what should have
  // been a bounce came out as a spread. going up is not cut, so the whole
  // animal travels. the small term the other way is the beat itself landing.
  const lift = Math.round(f.hop * 2.4 - f.pat * 0.7) + f.groove * (1 - f.bob) * 1.2 - f.bob * 0.4
  const breath = Math.sin(f.phase * (f.mood === "sleep" ? 0.7 : 1.15)) * 0.2
  tailFlick = Math.sin(f.phase * (1.2 + f.bob * 2.6)) * (0.7 + f.bob * 1.7)

  faceWide = wide
  const floor = GROUND - lift
  baseRow = Math.round(floor)
  const up = (d: number) => floor - (PIVOT + (d - PIVOT) * tall)
  const out = (d: number) => BODY_X + d * wide

  // the lean. this is most of what makes the cat feel like it is watching you:
  // pupils sliding about inside a head that never moves read as a glitch, a
  // whole head carried two dots towards the cursor reads as attention.
  const lean = f.gazeX * 2.0
  const nod = f.gazeY * 1.1
  // keeping time. the head rocks side to side and settles into the shoulders
  // on the beat, which is the whole of what a cat sitting in front of a
  // speaker does about it.
  //
  // the amplitude never drops below a dot. below that the rounding turns a
  // smooth sine into a coin toss - the head sits still for a while and then
  // jumps a dot for one frame, which reads as a glitch rather than as time
  // being kept.
  const sway = f.notes ? Math.sin(f.phase * 2.4) * (1.0 + f.bob * 1.4) : 0

  // ---- the tail. only the last third of it: the rest is behind the loaf,
  // which is where a cat sitting like this keeps it. it sweeps out along the
  // ground past the flank and then curls up, and the curl is the part that
  // moves - faster and further the louder the music.
  limb(
    BLOBS,
    out(11.0),
    up(1.4),
    out(21.0),
    up(1.0),
    out(19.5),
    up(8.0 + tailFlick),
    2.0 * rScale,
    0.85,
    8,
  )

  // ---- the loaf and the head, which are one mass.
  //
  // three goes at this drew a head sitting on a body, with a carve at the join
  // to keep the two apart, and every one of them came out a pyramid with ears
  // or a cat behind a box. a cat in a loaf has no neck and no shoulders to
  // speak of: it is a single rounded trapezoid, narrower where the face is and
  // wider where it meets the ground, and the only thing that says which part
  // is the head is the face drawn on it. so the ranks below run continuously
  // from skull to base, and the one job of the geometry is that the widening
  // happens over two rows rather than over eight - a slope that gradual is a
  // tent, and a tent is not an animal.
  const sag = Math.round(lean * 0.25)
  headR = 5.0 * rScale
  headX = Math.round(HEAD_X + lean + sway)
  // no beat term here. the head rides on `up`, which carries the lift, so it
  // already goes down and comes back with the rest of the animal - and a cat
  // bopping moves in one piece. a beat added on top of that sinks the head
  // into the shoulders instead, which is a cat being pressed rather than a cat
  // keeping time, and it costs the chest the three dots it has.
  headY = Math.round(up(14.0) + nod)
  const skull = 2.2 * wide + (f.mood === "happy" || f.mood === "cheer" ? 0.4 : 0)
  head(headX - skull, headY, headR)
  head(headX, headY, headR * 1.02)
  head(headX + skull, headY, headR)
  // jowls: a lower, wider pair. a cat's head is broadest at the cheek, and
  // these are also what the body's shoulders come up to meet.
  const jowl = 4.0 * wide
  head(headX - jowl, headY + 2.5, 4.0 * rScale)
  head(headX + jowl, headY + 2.5, 4.0 * rScale)

  // the body: three ranks, each a shade wider than the one above it, the top
  // one set high enough to catch the jowls so there is no waist between them
  // the back, walked front to rear: low at the shoulder where the head sits on
  // it, rising to the haunch, dropping away again at the tail end. a level
  // back is a bench; this one is the line that says which end the cat keeps
  // its legs under.
  const back: Array<[number, number]> = [
    [-11.0, 7.8],
    [-7.0, 8.8],
    [-3.0, 9.2],
    [1.0, 9.3],
    [5.0, 9.3],
    [9.0, 9.0],
    [12.0, 8.2],
  ]
  for (const [d, h] of back) BLOBS.add(out(d) + sag, up(h), (3.5 + breath) * rScale)
  // flanks and base stay square to the ground: a cat settled like this is
  // level underneath whatever its back is doing
  for (const d of [-12.0, -8.0, -4.0, 0, 4.0, 8.0, 12.0]) {
    BLOBS.add(out(d) + sag, up(4.5), 3.6 * rScale)
  }
  for (const d of [-11.8, -8.0, -4.0, 0, 4.0, 8.0, 11.8]) {
    BLOBS.add(out(d) + sag, up(1.0), 3.6 * rScale)
  }

  // ---- ears. how far they are pricked is more of the cat's mood at a glance
  // than the face is: up when pleased, folded out flat when asleep, and one of
  // them flicking whenever something happens.
  const mooded =
    f.mood === "cheer"
      ? 1
      : f.mood === "happy" || f.mood === "purr"
        ? 0.7
        : f.mood === "sleep"
          ? -1
          : f.mood === "focus"
            ? 0.5
            : f.affection * 0.5
  // a hand on the head folds them back whatever the mood says
  const perk = mooded - f.pat * 1.0
  ear(-1, perk, skull, wide, tall, rScale, f.twitch * 0.5)
  ear(1, perk, skull, wide, tall, rScale, f.twitch)
}

// a part of the head: into the whole cat, and again into the head on its own
const head = (x: number, y: number, r: number) => {
  BLOBS.add(x, y, r)
  HEAD_BLOBS.add(x, y, r)
}

// one ear, grown out of the skull as a tapering chain. side is -1 or 1.
function ear(
  side: number,
  perk: number,
  skullHalf: number,
  wide: number,
  tall: number,
  rScale: number,
  flick: number,
) {
  // drooping swings the tip out and down; perking stands it up and draws it in
  // the tips splay out to about the width of the skull whatever the mood is
  // doing to them. pulled in much further than that they leave a step at the
  // crown, and the head reads as a dome with two small horns on it.
  const spread = 3.2 - perk * 0.5 + flick * 0.9
  const height = (7.0 + perk * 0.8 - flick * 1.4) * tall
  const baseX = headX + side * (skullHalf + headR * 0.48)
  const baseY = headY - headR * 0.55
  // into both sets: the ears are part of the head, and an outline taken from a
  // head without them would draw a line straight across their base
  for (const set of [BLOBS, HEAD_BLOBS]) {
    limb(
      set,
      baseX,
      baseY,
      // the control point bows the outer edge, which is the difference between
      // a cat's ear and a traffic cone
      baseX + side * spread * 0.3,
      baseY - height * 0.6,
      baseX + side * spread * wide,
      baseY - height,
      2.7 * rScale,
      0.55,
      5,
    )
  }
}

// the head's own outline, struck onto the body wherever the two overlap.
//
// a loaf has no neck, so the head and the body are one field and the silhouette
// runs from ear to rump without a break in it - which leaves the head reading
// as the narrow end of a mound rather than as a head. drawing the head's edge
// where it lies over the chest and the shoulder puts the two masses back in
// front of and behind each other. it is the one line an artist would draw
// first and the only one a merged field cannot give you.
//
// the rim is taken from the head's field the same way the silhouette's is
// taken from the whole cat's, so the two are one continuous outline: outside,
// the head's edge already is the silhouette; inside, this is the rest of it.
function drawHeadEdge(g: Grid) {
  HEAD_BLOBS.scatter(HEAD_FIELD, PET_W, PET_H)
  if (baseRow + 1 < PET_H) HEAD_FIELD.fill(0, (baseRow + 1) * PET_W)
  for (let y = 1; y < PET_H - 1; y++) {
    const base = y * PET_W
    for (let x = 1; x < PET_W - 1; x++) {
      const i = base + x
      if (HEAD_FIELD[i] < SURFACE) continue
      if (
        HEAD_FIELD[i - 1] >= SURFACE &&
        HEAD_FIELD[i + 1] >= SURFACE &&
        HEAD_FIELD[i - PET_W] >= SURFACE &&
        HEAD_FIELD[i + PET_W] >= SURFACE
      ) {
        continue
      }
      // `onFur` and not `at`: where the head's edge already is the outside of
      // the cat it is drawn once, by shadeSolid, and must not be drawn twice
      onFur(g, y, x, HOT)
      // and a shadow in the fur immediately outside it. a lit line on its own
      // is a line drawn on a surface; a lit line with a dark one under it is
      // one surface standing off another, and that is the whole difference
      // between a face painted on a loaf and a head in front of a body.
      if (HEAD_FIELD[i - 1] < SURFACE) shade(g, y, x - 1)
      if (HEAD_FIELD[i + 1] < SURFACE) shade(g, y, x + 1)
      if (HEAD_FIELD[i - PET_W] < SURFACE) shade(g, y - 1, x)
      if (HEAD_FIELD[i + PET_W] < SURFACE) shade(g, y + 1, x)
    }
  }
}

// one dot of shadow, laid only on plain fur - never on the silhouette's own
// rim, which would notch it, and never on a dot the head's outline has already
// claimed.
//
// two ranks below the fur, not one. the two themes each lean on a different
// half of this pair: on the dark panel the outline is cream against amber and
// carries it, on paper the outline is near-black against dark brown and barely
// registers, so there the shadow has to. one rank down it did not.
const shade = (g: Grid, r: number, c: number) => onFur(g, r, c, DIM)

// the ground the loaf is sitting on, drawn as a shadow rather than as a line.
// it is here for the hop: without something staying put underneath, a cat that
// lifts two dots reads as a cat that has been nudged rather than as a cat that
// has jumped.
function drawShadow(g: Grid, f: PetFrame) {
  const r = Math.round(GROUND) + 1
  const half = Math.round(15 - f.hop * 6)
  const mid = Math.round(BODY_X)
  for (let c = mid - half; c <= mid + half; c++) {
    // thinned at the ends, so it reads as a pool rather than as a plank
    at(g, r, c, Math.abs(c - mid) > half - 2 ? FAINT : DIM)
  }
}

// ---------------------------------------------------------------------------
// markings struck onto the silhouette once it is shaded
// ---------------------------------------------------------------------------
function drawMarkings(g: Grid) {
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

  // the collar, sitting on the chest a clear rank below the jaw. it used to be
  // laid right along the jawline, where it was doing the separating; now that
  // the head carries its own outline the collar is free to be what it is, and
  // two bands stacked on the same three rows only read as one thick one.
  const collar = hr + 9
  overRow(g, collar, hc - 6, hc + 6, ACCENT)
  onFur(g, collar + 1, hc, HOT)

  // ---- the two front paws, tucked under the front of the loaf. they are the
  // detail that says loaf rather than lump: a cat sitting like this has its
  // paws folded away with just the toes out in front.
  // one clear of the base line: that bottom row is all rim, and a pale paw
  // struck onto the rim is a paw you cannot see
  const pr = baseRow - 2
  const pawX = hc + 1
  for (const side of [-1, 1]) {
    const near = pawX + side * 2
    const far = pawX + side * 6
    furRow(g, pr, Math.min(near, far), Math.max(near, far), HOT)
    furRow(g, pr + 1, Math.min(near, far), Math.max(near, far), HOT)
    // one toe split each. two would be truer and at four dots across it comes
    // out as a comb. struck with `at` rather than `onFur`, because by now the
    // paw it is splitting is no longer fur.
    at(g, pr, pawX + side * 4, MID_SHADE)
    at(g, pr + 1, pawX + side * 4, MID_SHADE)
  }

  // ---- the haunch: the curve of the hind leg folded up under the rear. it is
  // one line and it does more for the read than anything else on the body -
  // without it the back half is a featureless slab, and a cat is an animal
  // whose back half you can see the mechanics of even when it is asleep.
  const hx = Math.round(BODY_X) + 12
  const hy = baseRow + 1
  for (let i = 0; i <= 14; i++) {
    const a = (i / 14) * (Math.PI / 2)
    onFur(g, Math.round(hy - Math.sin(a) * 7.6), Math.round(hx - Math.cos(a) * 9), MID_SHADE)
  }

  // the chest bib that used to run from the collar down to the paws is gone.
  // with the jaw line, its shadow, the collar and the paws all stacked on the
  // same six rows of chest there was no room left for it to be a marking - it
  // was two pale dots in a crowd, which is noise.
}

function drawFace(g: Grid, f: PetFrame) {
  const beaming = f.mood === "happy" || f.mood === "cheer" || f.mood === "purr"
  const hc = Math.round(headX)
  const hr = Math.round(headY)

  // how far the lids have come down. the eyes are the loudest thing on the
  // panel, so most of the difference between one mood and the next is here.
  const lid = f.mood === "focus" ? 2 : f.mood === "bop" ? 1 : 0

  for (const side of [-1, 1]) {
    // the socket runs from three to six dots out. four across is the width
    // that still reads as an eye: at five the pupil has to grow with it and
    // the pair come out as two lit windows.
    const inner = hc + side * 3
    const outer = hc + side * 6
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
      // the pupil is three dots across inside a socket five across, so it has
      // a dot of travel each way - enough to read as a look when the head is
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

  // whiskers, struck onto the cheek. a loaf is as wide at the jaw as it is
  // anywhere, so there is no clear panel beside the muzzle to hang them in -
  // dashes floating out level with the widest part of the body read as fins.
  // two ranks below the fur, on the cheek where they grow, they read.
  for (const side of [-1, 1]) {
    const far = hc + side * 7
    const near = hc + side * 5
    furRow(g, my, Math.min(far, near), Math.max(far, near), DIM)
    furRow(g, my + 2, Math.min(far, near), Math.max(far, near), DIM)
  }

  // the flush of being noticed, warm on the cheeks under the eyes
  if (f.affection > 0.4 || beaming) {
    for (const side of [-1, 1]) {
      onFur(g, hr, hc + side * 6, MID_SHADE)
      onFur(g, hr, hc + side * 7, MID_SHADE)
      onFur(g, hr + 1, hc + side * 6, MID_SHADE)
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
      const c = hc + 10 + i * 4 + Math.round(Math.sin(rise * 4 + i) * 1.5)
      note(g, r, c, rise > 0.8 ? DIM : LIT)
    }
  }

  if (f.affection > 0) {
    for (let i = 0; i < 2; i++) {
      const rise = (f.phase * 1.1 + i * 0.5) % 1
      if (rise > f.affection) continue
      heart(g, Math.round(hr - 7 - rise * 5), hc - 12 - i * 3, rise > 0.7 ? DIM : ACCENT)
    }
  }

  if (f.mood === "sleep") {
    for (let i = 0; i < 3; i++) {
      const t = (f.phase * 0.5 + i * 0.33) % 1
      zed(g, Math.round(hr - 6 - t * 4) - i * 2, hc + 9 + i * 3, t > 0.75 ? DIM : LIT)
    }
  }
}
