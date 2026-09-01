// metaballs, struck onto a dot matrix.
//
// the panel draws in dots, and a dot grid is a poor place for straight-sided
// sprites: at this pitch every diagonal is a staircase and every limb is a
// rectangle with a corner on it. a scalar field has no such trouble. each blob
// is a soft hill of influence, the field is the sum of the hills, and every
// dot above the surface threshold lights - so two blobs near each other fuse
// into one flowing shape, an ear grows out of a skull instead of being glued
// to it, and a tail is a line of blobs that thins away to nothing. quantising
// that field onto the grid is what gives the shapes their wobble, and it is
// the only drawing primitive on the page that can be squashed and stretched
// per frame without falling apart.
//
// the falloff is Wyvill's, (1 - d²/R²)³. it reaches exactly zero at R, which
// is the whole reason it is here instead of the textbook 1/d²: a blob costs
// only the cells inside its own bounding box, and nothing on the far side of
// the panel is quietly tugged out of shape by something it cannot see.

// influence reaches twice as far as the radius asked for, and the surface sits
// at the value halfway out. the arithmetic lives here so the numbers in a
// scene mean something plain: `add(x, y, 6)` draws a disc of radius six, and
// two blobs fuse once they are closer than their radii add up to.
const REACH = 2

// the field value at the edge of a lone blob: (1 - 0.5²)³
export const SURFACE = 0.421875

// a set of blobs, gathered per frame and then sampled. the buffer is reused
// between frames - at thirty frames a second a fresh array per frame is a
// steady drip of garbage for no gain.
export class BlobSet {
  // four floats to a blob: centre x, centre y, influence radius, weight
  private data = new Float32Array(96 * 4)
  count = 0

  reset() {
    this.count = 0
  }

  // weight below zero carves instead of adding, which is how a shape gets a
  // notch - between two feet, say - without a second pass over the grid
  add(cx: number, cy: number, r: number, weight = 1) {
    if (r <= 0 || weight === 0) return
    const i = this.count * 4
    if (i + 4 > this.data.length) {
      const grown = new Float32Array(this.data.length * 2)
      grown.set(this.data)
      this.data = grown
    }
    this.data[i] = cx
    this.data[i + 1] = cy
    this.data[i + 2] = r * REACH
    this.data[i + 3] = weight
    this.count++
  }

  // the field at an arbitrary point, for callers whose dots are not on a grid
  at(x: number, y: number): number {
    const d = this.data
    const end = this.count * 4
    let sum = 0
    for (let i = 0; i < end; i += 4) {
      const dx = x - d[i]
      const dy = y - d[i + 1]
      const r2 = d[i + 2] * d[i + 2]
      const q = dx * dx + dy * dy
      if (q >= r2) continue
      const t = 1 - q / r2
      sum += d[i + 3] * t * t * t
    }
    return sum
  }

  // sum the blobs into a w×h grid, each one touching only the cells it reaches.
  // scattering by bounding box rather than asking every cell about every blob
  // is the difference between a creature costing tens of thousands of distance
  // checks a frame and costing a few thousand.
  scatter(field: Float32Array, w: number, h: number) {
    field.fill(0)
    const d = this.data
    const end = this.count * 4
    for (let i = 0; i < end; i += 4) {
      const cx = d[i]
      const cy = d[i + 1]
      const reach = d[i + 2]
      const k = d[i + 3]
      const r2 = reach * reach
      const x0 = Math.max(0, Math.ceil(cx - reach))
      const x1 = Math.min(w - 1, Math.floor(cx + reach))
      const y0 = Math.max(0, Math.ceil(cy - reach))
      const y1 = Math.min(h - 1, Math.floor(cy + reach))
      for (let y = y0; y <= y1; y++) {
        const dy = y - cy
        // how much of the radius budget is left for this row once the vertical
        // distance is paid for; a row past the blob's poles has none
        const spare = r2 - dy * dy
        if (spare <= 0) continue
        const base = y * w
        for (let x = x0; x <= x1; x++) {
          const dx = x - cx
          const q = dx * dx
          if (q >= spare) continue
          const t = (spare - q) / r2
          field[base + x] += k * t * t * t
        }
      }
    }
  }
}

// blobs walked along a quadratic curve with the radius easing base to tip. one
// call is an ear, a tail, or a leg; the control point is what lets a tail curl
// rather than merely point.
export function limb(
  b: BlobSet,
  x0: number,
  y0: number,
  cx: number,
  cy: number,
  x1: number,
  y1: number,
  rBase: number,
  rTip: number,
  steps = 5,
) {
  for (let i = 0; i <= steps; i++) {
    const t = i / steps
    const u = 1 - t
    b.add(
      u * u * x0 + 2 * u * t * cx + t * t * x1,
      u * u * y0 + 2 * u * t * cy + t * t * y1,
      rBase + (rTip - rBase) * t,
    )
  }
}

export interface SolidStyle {
  // the body of the shape
  fill: number
  // the one dot of outline around it
  rim: number
  // an optional glow in the dots just outside the surface
  halo?: number
  // where the halo starts, as a fraction of the surface value
  haloAt?: number
  // the unlit panel
  off?: number
  surface?: number
}

// a filled silhouette with a single dot of rim all round it.
//
// the rim is taken from the thresholded mask rather than from a band of field
// values, and that is not a detail: the field falls off far more steeply round
// a small blob than a large one, so a rim cut by value is two dots thick on a
// belly and missing entirely on an ear tip. taken from the mask it is exactly
// one dot everywhere, which is what makes a creature read as drawn rather than
// as fogged.
export function shadeSolid(
  field: Float32Array,
  out: Uint8Array,
  w: number,
  h: number,
  style: SolidStyle,
) {
  const surface = style.surface ?? SURFACE
  const off = style.off ?? 0
  const haloFrom = surface * (style.haloAt ?? 0.45)
  const halo = style.halo ?? off

  for (let i = 0; i < field.length; i++) {
    const v = field[i]
    out[i] = v >= surface ? style.fill : v >= haloFrom ? halo : off
  }

  for (let y = 0; y < h; y++) {
    const base = y * w
    for (let x = 0; x < w; x++) {
      const i = base + x
      if (field[i] < surface) continue
      // off the edge of the grid counts as empty, so a shape resting on the
      // bottom row still gets a line under it
      const edge =
        x === 0 ||
        x === w - 1 ||
        y === 0 ||
        y === h - 1 ||
        field[i - 1] < surface ||
        field[i + 1] < surface ||
        field[i - w] < surface ||
        field[i + w] < surface
      if (edge) out[i] = style.rim
    }
  }
}

// a shade per band of field value, brightest first. this is the other reading
// of the same field: instead of a silhouette with an outline, a soft mound
// that is hottest where the blobs pile up. good for a wave or an ambience,
// wrong for anything that has to be recognised as a thing.
export type Band = readonly [above: number, shade: number]

export function shadeRamp(field: Float32Array, out: Uint8Array, bands: readonly Band[], off = 0) {
  for (let i = 0; i < field.length; i++) {
    const v = field[i]
    let shade = off
    for (let b = 0; b < bands.length; b++) {
      if (v >= bands[b][0]) {
        shade = bands[b][1]
        break
      }
    }
    out[i] = shade
  }
}
