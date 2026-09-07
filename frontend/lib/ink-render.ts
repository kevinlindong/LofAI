// ink on a dot matrix: metaball dots that fuse into each other.
//
// the dots are dots again - circles on the lattice, the way the panel drew them
// before - and what is new is what happens BETWEEN two of them. a dot that
// grows until it reaches its neighbour does not simply touch it: the two pull a
// membrane between them that leaves each circle along its own tangent and
// pinches in the middle, so a pair reads as one poured shape with a waist
// rather than as two beads and a bar.
//
// the connector is not improvised here. it is the standard metaball connector:
//
//   Metaball script by Hiroyuki Sato    http://shspage.com/aijs/en/#metaball
//   ported to Paper.js by kynd          http://paperjs.org/examples/meta-balls
//   derivation written up by Varun Vachhar   https://varun.ca/metaballs/
//
// the geometry, in one paragraph, because the constants below are meaningless
// without it: the widest a membrane between two circles can be is bounded by
// the two external tangent lines common to both, and the angle to a tangent
// point is `acos((r1 - r2) / d)`. `spread` is how much of that maximum this
// membrane actually takes. the four points where the membrane meets the two
// circles are placed at those angles, and each gets a bezier handle turned a
// quarter turn off its own radius - which is to say, laid along the circle's
// tangent at that point. that tangency is the whole thing: it is what makes the
// membrane leave the circle smoothly instead of cornering into it, and it is
// why this reads as liquid where a hand-placed control point does not. the
// handle length falls off as the two circles separate, so a membrane thins and
// finally lets go rather than snapping.
//
// when the circles overlap, `u1`/`u2` open the spread out by the overlap angle
// (law of cosines) so the membrane stays outside the lens the two circles
// already share, and the whole thing collapses cleanly once one circle contains
// the other.
//
// each colour is one path, filled once with the nonzero rule, so circles and
// membranes union instead of seaming. that requires every subpath to wind the
// same way round - see `connector` for why it is traversed the way it is.

const TAU = Math.PI * 2

export interface InkGeometry {
  // radius of a cell holding the least ink that still counts as wet
  minRadius: number
  // radius of a saturated cell. at half the pitch two neighbours are exactly
  // tangent and past it they overlap, which is what lets a solid region fuse
  // into one mass rather than a grid of touching coins
  maxRadius: number
  // how much of the maximum tangent spread a membrane takes, 0..1. Sato's `v`.
  // 0.5 is the value the Paper.js port uses and it is a good one: lower makes
  // strings, higher makes the membrane bulge past the circles it joins
  spread: number
  // handle length factor. Sato's `handle_len_rate`, 2.4 upstream. this is the
  // dial for how taut or how slack the membrane looks
  handleSize: number
  // Maximum separation relative to the mean radius. Symmetric so reversing
  // a pair cannot change whether it joins. Sato's default is 2.5.
  reach: number
  // cells of the dry shade are struck at this radius and never fuse
  dryRadius: number
  // fuse diagonally when the two orthogonal cells between a diagonal pair are
  // both dry, so a diagonal run of ink flows instead of staircasing
  diagonals: boolean
  // below this much fill, the drawn radius fades to nothing, so a cell that
  // has only just wetted swells in from zero instead of popping in at
  // minRadius - and a cell that is drying shrinks away instead of blinking
  // out. 0 or absent keeps the old behaviour: a wet cell is at least
  // minRadius from its first frame.
  swellIn?: number
}

// the lattice, as flat arrays rather than as a rectangle: the visualiser's grid
// has a hole punched in the middle for the transport button, so a cell's
// neighbours cannot be worked out from its index.
//
// the neighbour arrays all point FORWARD - right, down, and the two diagonals
// below - so walking every cell once visits every pair exactly once. -1 is off
// the lattice.
export interface InkCells {
  count: number
  x: Float32Array
  y: Float32Array
  // 0 is dry and draws nothing; otherwise how much ink this cell holds, 0..1
  fill: Float32Array
  // which palette entry this cell belongs to. cells only ever fuse with
  // neighbours of the same shade, so each shade stays its own body - which is
  // what keeps a one-cell rim reading as an outline around a mass rather than
  // melting into it
  shade: Uint8Array
  right: Int32Array
  left: Int32Array
  down: Int32Array
  downRight: Int32Array
  downLeft: Int32Array
}

// the membrane between two circles.
//
// Sato's construction, with one change: upstream draws the connector plus the
// far side of the second circle, because upstream is not also drawing the
// circles. here the circles are already in the path, so the membrane closes
// across each circle's own chord instead - the chord between two points on a
// circle is inside it, so the union comes out identical and there is no arc to
// get the sweep flag wrong on.
//
// it is walked p2 -> p4 -> p3 -> p1 rather than Sato's p1 -> p3 -> p4 -> p2.
// the curve is the same either way, but this direction winds the same way as
// `arc` does, and under the nonzero rule two subpaths that wind oppositely
// cancel where they overlap - which would punch a hole through every join
// instead of filling it.
function connector(
  ctx: CanvasRenderingContext2D,
  x1: number,
  y1: number,
  r1: number,
  x2: number,
  y2: number,
  r2: number,
  geo: InkGeometry,
): boolean {
  if (r1 <= 0 || r2 <= 0) return false
  const dx = x2 - x1
  const dy = y2 - y1
  const d = Math.sqrt(dx * dx + dy * dy)
  // too far to hold a membrane at all, or one circle already contains the
  // other and there is nothing between them to draw
  const limit = (r1 + r2) * (1 + geo.reach) * 0.5
  if (geo.reach <= 0 || d >= limit) return false
  if (d <= Math.abs(r1 - r2)) return false

  // Let the neck close continuously before releasing it. The old hard
  // distance cutoff removed a finite-width bridge in a single frame.
  const total = r1 + r2
  const tension = Math.min(1, Math.max(0, (limit - d) / Math.max(0.001, limit - total)))
  const v = geo.spread * tension * tension * (3 - 2 * tension)
  let u1 = 0
  let u2 = 0
  if (d < r1 + r2) {
    // overlapping: open the spread out by the half-angle of the lens the two
    // circles share, so the membrane stays outside it
    u1 = Math.acos((r1 * r1 + d * d - r2 * r2) / (2 * r1 * d))
    u2 = Math.acos((r2 * r2 + d * d - r1 * r1) / (2 * r2 * d))
  }

  // the angle to the external tangent point: the widest the membrane can be
  const maxSpread = Math.acos((r1 - r2) / d)
  const a = u1 + (maxSpread - u1) * v
  const b = Math.PI - u2 - (Math.PI - u2 - maxSpread) * v
  const ux = dx / d
  const uy = dy / d
  const ca = Math.cos(a)
  const sa = Math.sin(a)
  const cb = Math.cos(b)
  const sb = Math.sin(b)
  // Work in the pair's local basis: four trig calls instead of sixteen,
  // with tangent handles obtained by rotating each radius a quarter turn.
  const n1x = ux * ca - uy * sa
  const n1y = uy * ca + ux * sa
  const n2x = ux * ca + uy * sa
  const n2y = uy * ca - ux * sa
  const n3x = ux * cb - uy * sb
  const n3y = uy * cb + ux * sb
  const n4x = ux * cb + uy * sb
  const n4y = uy * cb - ux * sb
  const p1x = x1 + n1x * r1
  const p1y = y1 + n1y * r1
  const p2x = x1 + n2x * r1
  const p2y = y1 + n2y * r1
  const p3x = x2 + n3x * r2
  const p3y = y2 + n3y * r2
  const p4x = x2 + n4x * r2
  const p4y = y2 + n4y * r2

  // handle length, shortening as the circles separate and again as they overlap
  const chord = Math.sqrt((p3x - p1x) * (p3x - p1x) + (p3y - p1y) * (p3y - p1y))
  // Keep both handles on their own side of the axis as the spread closes.
  // Without this bound a thin releasing neck can fold through itself.
  const neckLimit = Math.min(ca > 0 ? sa / ca : Infinity, cb < 0 ? -sb / cb : Infinity) * 0.98
  const d2 = Math.min(v * geo.handleSize, chord / total, neckLimit) * Math.min(1, (d * 2) / total)
  const h1len = r1 * d2
  const h2len = r2 * d2

  // a quarter turn off each point's own radius is that circle's tangent there.
  // this is the line the membrane has to leave along, and the reason the join
  // reads as liquid rather than as a shape glued on.
  const h1x = p1x + n1y * h1len
  const h1y = p1y - n1x * h1len
  const h2x = p2x - n2y * h1len
  const h2y = p2y + n2x * h1len
  const h3x = p3x - n3y * h2len
  const h3y = p3y + n3x * h2len
  const h4x = p4x + n4y * h2len
  const h4y = p4y - n4x * h2len

  ctx.moveTo(p2x, p2y)
  ctx.bezierCurveTo(h2x, h2y, h4x, h4y, p4x, p4y)
  ctx.lineTo(p3x, p3y)
  ctx.bezierCurveTo(h3x, h3y, h1x, h1y, p1x, p1y)
  ctx.closePath()
  return true
}

// can two circles this far apart hold a membrane at all
interface InkScratch {
  radius: Float32Array
  bonded: Uint8Array
  square: Uint8Array
  up: Int32Array
}

const scratchByCells = new WeakMap<InkCells, InkScratch>()

function scratchFor(cells: InkCells): InkScratch {
  let scratch = scratchByCells.get(cells)
  if (!scratch || scratch.radius.length !== cells.count) {
    scratch = {
      radius: new Float32Array(cells.count),
      bonded: new Uint8Array(cells.count),
      square: new Uint8Array(cells.count),
      up: new Int32Array(cells.count).fill(-1),
    }
    for (let i = 0; i < cells.count; i++) {
      if (cells.down[i] >= 0) scratch.up[cells.down[i]] = i
    }
    scratchByCells.set(cells, scratch)
  }
  return scratch
}

// paint every shade of a lattice. `dryShade` is drawn as plain small circles
// that never fuse - it is the unlit panel and the holes cut in a body, and
// fusing it would run the background together into one sheet.
export function paintInk(
  ctx: CanvasRenderingContext2D,
  cells: InkCells,
  geo: InkGeometry,
  palette: readonly string[],
  dryShade = -1,
) {
  const { count, x, y, fill, shade, right, left, down, downRight, downLeft } = cells
  const { radius, bonded, square, up } = scratchFor(cells)
  const span = geo.maxRadius - geo.minRadius
  const swell = geo.swellIn ?? 0
  square.fill(0)

  // One radius calculation per cell, shared by every neighbour and shade.
  for (let i = 0; i < count; i++) {
    const f = Math.max(0, Math.min(1, fill[i]))
    const t = swell > 0 ? Math.min(1, f / swell) : 1
    radius[i] = shade[i] === dryShade ? geo.dryRadius :
      (geo.minRadius + span * f) * t * t * (3 - 2 * t)
    bonded[i] = f > 0 && f >= swell && shade[i] !== dryShade ? 1 : 0
  }

  // A square entirely inside overlapping dots is solid. Its internal circles
  // and connectors cannot affect the silhouette, so rasterise it as a strip.
  // Require actual overlap, not just a tenuous neck: otherwise filling the
  // centre turns four separated beads into a rectangle in a single frame.
  if (geo.reach > 0) {
    for (let i = 0; i < count; i++) {
      const r = right[i], d = down[i], dr = downRight[i]
      if (!bonded[i] || r < 0 || d < 0 || dr < 0 ||
          !bonded[r] || !bonded[d] || !bonded[dr] ||
          shade[r] !== shade[i] || shade[d] !== shade[i] || shade[dr] !== shade[i]) continue
      const dx = x[r] - x[i], dy = y[d] - y[i]
      // Only axis-aligned lattice squares. Subpixel-moving face details use
      // the ordinary path geometry and never get flattened into a strip.
      if (dx <= 0 || dy <= 0 || y[r] !== y[i] || x[d] !== x[i] ||
          x[dr] !== x[r] || y[dr] !== y[d]) continue
      const min = Math.min(radius[i], radius[r], radius[d], radius[dr])
      const max = Math.max(radius[i], radius[r], radius[d], radius[dr])
      if (min * 2 >= Math.max(dx, dy) && max <= Math.min(dx, dy)) square[i] = 1
    }
  }

  for (let s = 0; s < palette.length; s++) {
    let opened = false
    const dry = s === dryShade
    const same = (i: number) => i >= 0 && shade[i] === s && bonded[i] !== 0

    for (let i = 0; i < count; i++) {
      if (shade[i] !== s || fill[i] <= 0 || radius[i] <= 0) continue
      if (!opened) {
        ctx.beginPath()
        opened = true
      }

      const r = radius[i]
      const l = left[i], u = up[i]
      const ul = u >= 0 ? left[u] : -1
      const covered = square[i] && l >= 0 && square[l] &&
        u >= 0 && square[u] && ul >= 0 && square[ul]
      if (!covered) {
        ctx.moveTo(x[i] + r, y[i])
        ctx.arc(x[i], y[i], r, 0, TAU)
      }
      if (dry || geo.reach <= 0) continue

      if (bonded[i]) {
        for (let dir = 0; dir < (geo.diagonals ? 4 : 2); dir++) {
          const j = dir === 0 ? right[i] : dir === 1 ? down[i] :
            dir === 2 ? downRight[i] : downLeft[i]
          if (!same(j)) continue
          if (dir === 0 && square[i] && u >= 0 && square[u]) continue
          if (dir === 1 && square[i] && l >= 0 && square[l]) continue
          if (dir >= 2 && (same(dir === 2 ? right[i] : l) || same(down[i]))) continue
          connector(ctx, x[i], y[i], r, x[j], y[j], radius[j], geo)
        }
      }

      // Adjacent full squares share one rectangle, with the same winding as
      // the circles. No overlapping interior paths for Canvas to tessellate.
      if (square[i] && (l < 0 || !square[l])) {
        let end = i
        while (right[end] >= 0 && square[right[end]]) end = right[end]
        const rr = right[end], dd = down[i]
        ctx.moveTo(x[i], y[i])
        ctx.lineTo(x[rr], y[i])
        ctx.lineTo(x[rr], y[dd])
        ctx.lineTo(x[i], y[dd])
        ctx.closePath()
      }
    }

    if (opened) {
      ctx.fillStyle = palette[s]
      ctx.fill()
    }
  }
}

// neighbour tables for a plain w×h grid, where index is r*w+c and every cell
// exists. built once by the caller and handed back in `InkCells`.
export function gridNeighbours(w: number, h: number) {
  const n = w * h
  const right = new Int32Array(n)
  const left = new Int32Array(n)
  const down = new Int32Array(n)
  const downRight = new Int32Array(n)
  const downLeft = new Int32Array(n)
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const i = r * w + c
      const hasR = c + 1 < w
      const hasL = c > 0
      const hasD = r + 1 < h
      right[i] = hasR ? i + 1 : -1
      left[i] = hasL ? i - 1 : -1
      down[i] = hasD ? i + w : -1
      downRight[i] = hasD && hasR ? i + w + 1 : -1
      downLeft[i] = hasD && hasL ? i + w - 1 : -1
    }
  }
  return { right, left, down, downRight, downLeft }
}

// exported for the geometry tests, which need to reason about one membrane in
// isolation rather than about a whole lattice
export const __connector = connector
