// Each matrix cell is a compact density field, not a circle with a bridge
// pasted across the gap. Adding the fields stretches their facing surfaces
// before contact, opens a neck, and gradually closes the holes between cells.
// Four samples per pitch plus curved interpolation resolve the neck; canvas size and
// device pixel ratio never increase the simulation's work.
const SAMPLES = 4
// A little extra reach lets one droplet hand ink to the next even while it
// recedes. Calibrate the isovalue so an isolated droplet keeps its radius.
const REACH = 2.25
const SURFACE = (1 - 1 / (REACH * REACH)) ** 3

interface Lattice {
  count: number
  x: Float32Array
  y: Float32Array
}

export interface LiquidStyle {
  // Radius of a full droplet, as a fraction of the lattice pitch.
  radius?: number
  // Seconds for ink to enter/leave a cell. The surface retains a little ink
  // on release, so a retreating edge stretches before it separates.
  attack?: number
  release?: number
}

export class LiquidInk {
  private readonly width: number
  private readonly height: number
  private readonly originX: number
  private readonly originY: number
  private readonly step: number
  private readonly x: Float32Array
  private readonly y: Float32Array
  private readonly volume: Float32Array
  private readonly field: Float32Array
  private readonly gradientX: Float32Array
  private readonly gradientY: Float32Array
  private readonly next: Int32Array
  private readonly vertices: Int32Array
  private readonly vx: Float32Array
  private readonly vy: Float32Array
  private readonly tx: Float32Array
  private readonly ty: Float32Array
  private readonly radius: number
  private readonly attack: number
  private readonly release: number
  private initialized = false

  constructor(cells: Lattice, pitch: number, style: LiquidStyle = {}) {
    this.radius = style.radius ?? 0.54
    this.attack = style.attack ?? 0.075
    this.release = style.release ?? 0.12
    this.step = pitch / SAMPLES
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
    for (let i = 0; i < cells.count; i++) {
      minX = Math.min(minX, cells.x[i])
      minY = Math.min(minY, cells.y[i])
      maxX = Math.max(maxX, cells.x[i])
      maxY = Math.max(maxY, cells.y[i])
    }
    if (!cells.count) minX = minY = maxX = maxY = 0
    // A dry border closes every contour, including shapes touching the panel
    // edge. The callers' normal canvas bounds supply the final crop.
    const pad = Math.ceil(this.radius * REACH * SAMPLES) + 1
    this.originX = minX - pad * this.step
    this.originY = minY - pad * this.step
    this.width = Math.ceil((maxX - minX) / this.step) + pad * 2 + 1
    this.height = Math.ceil((maxY - minY) / this.step) + pad * 2 + 1
    this.x = new Float32Array(cells.count)
    this.y = new Float32Array(cells.count)
    this.volume = new Float32Array(cells.count)
    for (let i = 0; i < cells.count; i++) {
      this.x[i] = (cells.x[i] - this.originX) / this.step
      this.y[i] = (cells.y[i] - this.originY) / this.step
    }
    const n = this.width * this.height
    this.field = new Float32Array(n)
    this.gradientX = new Float32Array(n)
    this.gradientY = new Float32Array(n)
    // Two possible contour vertices per sample: its right and bottom edges.
    // Each crossing has exactly one outgoing link; holes wind the other way.
    this.next = new Int32Array(n * 2)
    this.vertices = new Int32Array(n * 2)
    this.vx = new Float32Array(n * 2)
    this.vy = new Float32Array(n * 2)
    this.tx = new Float32Array(n * 2)
    this.ty = new Float32Array(n * 2)
  }

  paint(ctx: CanvasRenderingContext2D, target: Float32Array, color: string, dt = 0) {
    const { field, gradientX, gradientY, width: w, height: h, volume } = this
    field.fill(0)
    gradientX.fill(0)
    gradientY.fill(0)
    const settle = !this.initialized || dt <= 0
    this.initialized = true
    const rise = settle ? 1 : 1 - Math.exp(-dt / this.attack)
    const fall = settle ? 1 : 1 - Math.exp(-dt / this.release)
    const fullReach = this.radius * REACH * SAMPLES

    for (let i = 0; i < volume.length; i++) {
      const want = Math.max(0, Math.min(1, target[i]))
      const f = volume[i] + (want - volume[i]) * (want > volume[i] ? rise : fall)
      volume[i] = f
      if (f < 0.0001) continue
      // Ink amount controls area. There is no minimum-radius jump or separate
      // threshold that turns a connection on after its dots have already grown.
      const reachSquared = fullReach * fullReach * f
      const r2 = Math.max(1, reachSquared)
      const reach = Math.sqrt(r2)
      // Filter sub-sample droplets by their area; a tiny droplet must not
      // become a full-strength sample merely because it is centered on one.
      const weight = Math.min(1, reachSquared)
      const cx = this.x[i], cy = this.y[i]
      const x0 = Math.max(0, Math.ceil(cx - reach))
      const x1 = Math.min(w - 1, Math.floor(cx + reach))
      const y0 = Math.max(0, Math.ceil(cy - reach))
      const y1 = Math.min(h - 1, Math.floor(cy + reach))
      const inverse = 1 / r2
      for (let y = y0; y <= y1; y++) {
        const dy = y - cy
        const spare = r2 - dy * dy
        const base = y * w
        for (let x = x0; x <= x1; x++) {
          const dx = x - cx
          const t = (spare - dx * dx) * inverse
          if (t > 0) {
            const strength = weight * t * t
            const slope = -6 * strength * inverse
            field[base + x] += strength * t
            gradientX[base + x] += slope * dx
            gradientY[base + x] += slope * dy
          }
        }
      }
    }

    const { next, vertices, vx, vy, tx, ty } = this
    next.fill(-1)
    let crossings = 0
    const link = (from: number, to: number) => {
      next[from] = to
      vertices[crossings++] = from
      const k = from >> 1
      const vertical = from & 1
      const stride = vertical ? w : 1
      const a = field[k], b = field[k + stride], delta = b - a
      // A monotone Hermite crossing follows the field's curvature without a
      // denser grid. Linear crossings visibly facet a growing droplet.
      const gradient = vertical ? gradientY : gradientX
      const m0 = delta * Math.max(0, Math.min(3, gradient[k] / delta))
      const m1 = delta * Math.max(0, Math.min(3, gradient[k + stride] / delta))
      const c3 = m0 + m1 - 2 * delta, c2 = 3 * delta - 2 * m0 - m1
      let t = (SURFACE - a) / delta
      for (let j = 0; j < 2; j++) {
        const slope = (3 * c3 * t + 2 * c2) * t + m0
        if (Math.abs(slope) > 1e-8) {
          t = Math.max(0, Math.min(1, t - (((c3 * t + c2) * t + m0) * t + a - SURFACE) / slope))
        }
      }
      vx[from] = this.originX + (k % w + (vertical ? 0 : t)) * this.step
      vy[from] = this.originY + (Math.floor(k / w) + (vertical ? t : 0)) * this.step
      const cross = vertical ? gradientX : gradientY
      const along = (3 * c3 * t + 2 * c2) * t + m0
      const across = cross[k] * (1 - t) + cross[k + stride] * t
      const dx = vertical ? across : along, dy = vertical ? along : across
      const length = Math.sqrt(dx * dx + dy * dy) || 1
      tx[from] = dy / length
      ty[from] = -dx / length
    }

    for (let y = 0; y < h - 1; y++) {
      for (let x = 0; x < w - 1; x++) {
        const i = y * w + x
        const a = field[i] - SURFACE, b = field[i + 1] - SURFACE
        const c = field[i + w + 1] - SURFACE, d = field[i + w] - SURFACE
        const mask = (a >= 0 ? 1 : 0) | (b >= 0 ? 2 : 0) | (c >= 0 ? 4 : 0) | (d >= 0 ? 8 : 0)
        if (mask === 0 || mask === 15) continue
        const top = i * 2, right = top + 3, bottom = top + w * 2, left = top + 1
        switch (mask) {
          case 1: link(top, left); break
          case 2: link(right, top); break
          case 3: link(right, left); break
          case 4: link(bottom, right); break
          case 5:
            // Resolve a saddle from its bilinear field, not a fixed diagonal.
            if (a * c > b * d) { link(top, right); link(bottom, left) }
            else { link(top, left); link(bottom, right) }
            break
          case 6: link(bottom, top); break
          case 7: link(bottom, left); break
          case 8: link(left, bottom); break
          case 9: link(top, bottom); break
          case 10:
            if (a * c < b * d) { link(left, top); link(right, bottom) }
            else { link(right, top); link(left, bottom) }
            break
          case 11: link(right, bottom); break
          case 12: link(left, right); break
          case 13: link(top, right); break
          case 14: link(left, top); break
        }
      }
    }

    if (!crossings) return
    ctx.beginPath()
    for (let i = 0; i < crossings; i++) {
      const first = vertices[i]
      if (next[first] < 0) continue
      let current = first
      ctx.moveTo(vx[first], vy[first])
      // Trace only the surface. Solid regions need no circles, connectors,
      // or square plugs. Tangents follow the density gradient, so even a
      // tiny four-vertex droplet stays round instead of becoming a diamond.
      while (next[current] >= 0) {
        const after = next[current]
        const dx = vx[after] - vx[current], dy = vy[after] - vy[current]
        const chord = Math.sqrt(dx * dx + dy * dy)
        const cosine = Math.max(-1, Math.min(1, tx[current] * tx[after] + ty[current] * ty[after]))
        const handle = chord * (2 / 3) / (1 + Math.sqrt((1 + cosine) * 0.5))
        const h0 = tx[current] * dx + ty[current] * dy > 0 ? handle : 0
        const h1 = tx[after] * dx + ty[after] * dy > 0 ? handle : 0
        ctx.bezierCurveTo(vx[current] + tx[current] * h0, vy[current] + ty[current] * h0,
          vx[after] - tx[after] * h1, vy[after] - ty[after] * h1, vx[after], vy[after])
        next[current] = -1
        current = after
      }
      ctx.closePath()
    }
    ctx.fillStyle = color
    ctx.fill()
  }
}
