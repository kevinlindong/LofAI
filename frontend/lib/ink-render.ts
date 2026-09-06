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
const HALF_PI = Math.PI / 2

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
  // how far apart two circles can be and still hold a membrane, as a multiple
  // of the far radius added to the near one. Sato's `maxDist` uses 2.5
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
  if (d > r1 + r2 * geo.reach) return false
  if (d <= Math.abs(r1 - r2)) return false

  const v = geo.spread
  let u1 = 0
  let u2 = 0
  if (d < r1 + r2) {
    // overlapping: open the spread out by the half-angle of the lens the two
    // circles share, so the membrane stays outside it
    u1 = Math.acos((r1 * r1 + d * d - r2 * r2) / (2 * r1 * d))
    u2 = Math.acos((r2 * r2 + d * d - r1 * r1) / (2 * r2 * d))
  }

  const between = Math.atan2(dy, dx)
  // the angle to the external tangent point: the widest the membrane can be
  const maxSpread = Math.acos((r1 - r2) / d)
  const a1 = between + u1 + (maxSpread - u1) * v
  const a2 = between - u1 - (maxSpread - u1) * v
  const a3 = between + Math.PI - u2 - (Math.PI - u2 - maxSpread) * v
  const a4 = between - Math.PI + u2 + (Math.PI - u2 - maxSpread) * v

  const p1x = x1 + Math.cos(a1) * r1
  const p1y = y1 + Math.sin(a1) * r1
  const p2x = x1 + Math.cos(a2) * r1
  const p2y = y1 + Math.sin(a2) * r1
  const p3x = x2 + Math.cos(a3) * r2
  const p3y = y2 + Math.sin(a3) * r2
  const p4x = x2 + Math.cos(a4) * r2
  const p4y = y2 + Math.sin(a4) * r2

  // handle length, shortening as the circles separate and again as they overlap
  const total = r1 + r2
  const chord = Math.sqrt((p3x - p1x) * (p3x - p1x) + (p3y - p1y) * (p3y - p1y))
  const d2 = Math.min(v * geo.handleSize, chord / total) * Math.min(1, (d * 2) / total)
  const h1len = r1 * d2
  const h2len = r2 * d2

  // a quarter turn off each point's own radius is that circle's tangent there.
  // this is the line the membrane has to leave along, and the reason the join
  // reads as liquid rather than as a shape glued on.
  const h1x = p1x + Math.cos(a1 - HALF_PI) * h1len
  const h1y = p1y + Math.sin(a1 - HALF_PI) * h1len
  const h2x = p2x + Math.cos(a2 + HALF_PI) * h1len
  const h2y = p2y + Math.sin(a2 + HALF_PI) * h1len
  const h3x = p3x + Math.cos(a3 + HALF_PI) * h2len
  const h3y = p3y + Math.sin(a3 + HALF_PI) * h2len
  const h4x = p4x + Math.cos(a4 - HALF_PI) * h2len
  const h4y = p4y + Math.sin(a4 - HALF_PI) * h2len

  ctx.moveTo(p2x, p2y)
  ctx.bezierCurveTo(h2x, h2y, h4x, h4y, p4x, p4y)
  ctx.lineTo(p3x, p3y)
  ctx.bezierCurveTo(h3x, h3y, h1x, h1y, p1x, p1y)
  ctx.closePath()
  return true
}

// can two circles this far apart hold a membrane at all
function reaches(r1: number, r2: number, d: number, reach: number): boolean {
  return r1 > 0 && r2 > 0 && d <= r1 + r2 * reach && d > Math.abs(r1 - r2)
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
  const span = geo.maxRadius - geo.minRadius
  const swell = geo.swellIn ?? 0
  const radiusOf = (i: number) => {
    const r = geo.minRadius + span * fill[i]
    if (swell <= 0 || fill[i] >= swell) return r
    // eased, not linear, so the growth arrives without a corner at either end
    const t = fill[i] / swell
    return r * t * t * (3 - 2 * t)
  }
  const gap = (a: number, b: number) => Math.hypot(x[b] - x[a], y[b] - y[a])

  for (let s = 0; s < palette.length; s++) {
    let opened = false
    const dry = s === dryShade
    const wet = (i: number) => i >= 0 && shade[i] === s && fill[i] > 0
    // a membrane needs a dot on both ends. a cell still swelling in has next
    // to no radius, and a membrane attached to it converges to a point - a
    // hair sticking out of the mass rather than a neck within it. so while a
    // cell is arriving or leaving it is only a bead, and it bonds once it
    // holds real ink.
    const bonded = (i: number) => wet(i) && fill[i] >= swell

    for (let i = 0; i < count; i++) {
      if (shade[i] !== s || fill[i] <= 0) continue
      if (!opened) {
        ctx.beginPath()
        opened = true
      }

      const r = dry ? geo.dryRadius : radiusOf(i)
      ctx.moveTo(x[i] + r, y[i])
      ctx.arc(x[i], y[i], r, 0, TAU)
      if (dry) continue

      // the four forward neighbours. walking forward only means each pair is
      // visited exactly once, so no membrane is ever drawn twice.
      if (bonded(i)) {
        for (let dir = 0; dir < 4; dir++) {
          const j =
            dir === 0 ? right[i] : dir === 1 ? down[i] : dir === 2 ? downRight[i] : downLeft[i]
          if (!bonded(j)) continue

          if (dir >= 2) {
            if (!geo.diagonals) continue
            // a diagonal membrane only where the L-shaped route is not already
            // ink. drawn regardless, it would pack the corners of a solid block
            // and the body would read as tiled rather than poured.
            if (bonded(dir === 2 ? right[i] : left[i])) continue
            if (bonded(down[i])) continue
          }

          connector(ctx, x[i], y[i], r, x[j], y[j], radiusOf(j), geo)
        }
      }

      // four circles meeting at the corners of one lattice square cover its
      // edges - the membranes see to that - but not its middle, and the pinhole
      // left behind reads as a perforation rather than as poured ink. the square
      // joining the four centres plugs it exactly: its corners are the centres,
      // which are deep inside their own circles, and its edges lie along the
      // centre-to-centre axes, which every membrane straddles. so it can only
      // ever add area that is already inside the union, and never changes the
      // silhouette.
      //
      // only for a square that is wet on all four corners and joined all the
      // way round. one corner too faint to hold a membrane means the square has
      // an open side, and plugging it then WOULD show.
      const rr = right[i]
      const dd = down[i]
      const dr = downRight[i]
      if (bonded(i) && bonded(rr) && bonded(dd) && bonded(dr)) {
        const rRight = radiusOf(rr)
        const rDown = radiusOf(dd)
        const rDiag = radiusOf(dr)
        if (
          reaches(r, rRight, gap(i, rr), geo.reach) &&
          reaches(r, rDown, gap(i, dd), geo.reach) &&
          reaches(rRight, rDiag, gap(rr, dr), geo.reach) &&
          reaches(rDown, rDiag, gap(dd, dr), geo.reach)
        ) {
          // wound the same way as `arc` sweeps, like every other subpath here
          ctx.moveTo(x[i], y[i])
          ctx.lineTo(x[rr], y[rr])
          ctx.lineTo(x[dr], y[dr])
          ctx.lineTo(x[dd], y[dd])
          ctx.closePath()
        }
      }
    }

    if (!opened) continue
    ctx.fillStyle = palette[s]
    ctx.fill()
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
