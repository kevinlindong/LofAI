// Geometry tests for the liquid ink renderer, evaluated from the TypeScript
// source against a stub 2D context that records path calls instead of
// rasterising them.
//
// The renderer uses Hiroyuki Sato's metaball connector (see lib/ink-render.ts
// for provenance), so most of what is worth pinning down is that the connector
// is wired up correctly and that the lattice logic around it agrees with the
// look it is meant to produce:
//
//   - the handles lie along each circle's TANGENT at its contact point. That
//     tangency is what makes the join read as liquid, and it is the one property
//     a hand-placed control point gets wrong.
//   - every subpath winds the same way, or the nonzero fill would cancel where
//     a connector overlaps its circles and punch a hole through each join.
//   - a dot too far from its neighbour holds no membrane, so ink joins up as it
//     grows rather than being wired together from the start.

import { readFileSync } from "node:fs"
import { createRequire } from "node:module"
import { fileURLToPath } from "node:url"
import path from "node:path"
import vm from "node:vm"

const require = createRequire(import.meta.url)
const ts = require("typescript")
const here = path.dirname(fileURLToPath(import.meta.url))
const source = readFileSync(path.join(here, "..", "lib", "ink-render.ts"), "utf8")
const compiled = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.CommonJS,
    target: ts.ScriptTarget.ES2020,
  },
}).outputText

const module_ = { exports: {} }
vm.runInNewContext(compiled, { module: module_, exports: module_.exports, Math, console })
const { paintInk, gridNeighbours, __connector: connector } = module_.exports

// a stub 2D context that records path calls instead of rasterising them
function stub() {
  const ops = []
  return {
    ops,
    fillStyle: "",
    beginPath() {
      ops.push({ op: "begin" })
    },
    moveTo(x, y) {
      ops.push({ op: "moveTo", x, y })
    },
    lineTo(x, y) {
      ops.push({ op: "lineTo", x, y })
    },
    arc(x, y, r, a0, a1, ccw) {
      ops.push({ op: "arc", x, y, r, a0, a1, ccw: !!ccw })
    },
    bezierCurveTo(c1x, c1y, c2x, c2y, x, y) {
      ops.push({ op: "bezier", c1x, c1y, c2x, c2y, x, y })
    },
    closePath() {
      ops.push({ op: "close" })
    },
    fill() {
      ops.push({ op: "fill", style: this.fillStyle })
    },
  }
}

const PITCH = 10
const GEO = {
  minRadius: 0.22 * PITCH,
  maxRadius: 0.54 * PITCH,
  spread: 0.5,
  handleSize: 2.4,
  reach: 2.5,
  dryRadius: 2,
  diagonals: true,
}
const PALETTE = ["#dry", "#a", "#b"]

function lattice(w, h, shade, fill) {
  const n = w * h
  const x = new Float32Array(n)
  const y = new Float32Array(n)
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      x[r * w + c] = c * PITCH + PITCH / 2
      y[r * w + c] = r * PITCH + PITCH / 2
    }
  }
  return {
    count: n,
    x,
    y,
    fill: Float32Array.from(fill),
    shade: Uint8Array.from(shade),
    ...gridNeighbours(w, h),
  }
}

let failures = 0
const check = (name, cond, detail = "") => {
  if (cond) {
    console.log(`PASS  ${name}`)
  } else {
    failures++
    console.log(`FAIL  ${name}${detail ? ` - ${detail}` : ""}`)
  }
}
const near = (a, b, tol = 1e-6) => Math.abs(a - b) <= tol

// membranes are the subpaths containing beziers
const membranes = (ops) => ops.filter((o) => o.op === "bezier")

// ---- 1. the membrane's handles lie along each circle's tangent ----
{
  // two equal circles, horizontally apart, not overlapping
  const ctx = stub()
  const r = 4
  const drew = connector(ctx, 0, 0, r, 12, 0, r, GEO)
  check("a membrane is drawn between two reachable circles", drew === true)

  // walk the recorded path: moveTo p2, bezier -> p4, lineTo p3, bezier -> p1
  const move = ctx.ops.find((o) => o.op === "moveTo")
  const bez = membranes(ctx.ops)
  const p2 = { x: move.x, y: move.y }
  const p4 = { x: bez[0].x, y: bez[0].y }
  const p1 = { x: bez[1].x, y: bez[1].y }

  // every contact point sits exactly on the circle it belongs to
  const onC1 = (p) => near(Math.hypot(p.x, p.y), r, 1e-9)
  const onC2 = (p) => near(Math.hypot(p.x - 12, p.y), r, 1e-9)
  check("contact points sit on their circles", onC1(p1) && onC1(p2) && onC2(p4),
    `|p1|=${Math.hypot(p1.x, p1.y).toFixed(6)} |p2|=${Math.hypot(p2.x, p2.y).toFixed(6)}`)

  // the handle out of p2 must be perpendicular to p2's radius: the tangent.
  // dot(radius, handle) == 0 is exactly that condition.
  const h2 = { x: bez[0].c1x - p2.x, y: bez[0].c1y - p2.y }
  const radial2 = { x: p2.x, y: p2.y }
  const dot = radial2.x * h2.x + radial2.y * h2.y
  const scale = Math.hypot(radial2.x, radial2.y) * Math.hypot(h2.x, h2.y)
  check("the handle lies along the circle's tangent", near(dot / scale, 0, 1e-9),
    `cos(angle) = ${(dot / scale).toExponential(2)}`)

  // and the handle into p4 is tangent to the second circle
  const h4 = { x: bez[0].c2x - p4.x, y: bez[0].c2y - p4.y }
  const radial4 = { x: p4.x - 12, y: p4.y }
  const dot4 = radial4.x * h4.x + radial4.y * h4.y
  const scale4 = Math.hypot(radial4.x, radial4.y) * Math.hypot(h4.x, h4.y)
  check("the far handle is tangent to the far circle", near(dot4 / scale4, 0, 1e-9),
    `cos(angle) = ${(dot4 / scale4).toExponential(2)}`)
}

// ---- 2. the membrane is concave: it pinches between the circles ----
{
  const ctx = stub()
  const r = 4
  connector(ctx, 0, 0, r, 12, 0, r, GEO)
  const bez = membranes(ctx.ops)[0]
  const move = ctx.ops.find((o) => o.op === "moveTo")
  // this membrane runs along the top (negative y). sample the cubic at its
  // midpoint and compare with the straight chord between its endpoints.
  const at = (t) => {
    const u = 1 - t
    return (
      u * u * u * move.y +
      3 * u * u * t * bez.c1y +
      3 * u * t * t * bez.c2y +
      t * t * t * bez.y
    )
  }
  const mid = at(0.5)
  const chord = (move.y + bez.y) / 2
  // pinched means the curve's waist is CLOSER to the axis than the chord is
  check("the membrane pinches inward at its waist", mid > chord,
    `waist y=${mid.toFixed(3)} vs chord y=${chord.toFixed(3)}`)
  check("the waist stays outside the axis", mid < 0,
    `waist y=${mid.toFixed(3)}`)
}

// ---- 3. reach: too far apart and there is no membrane at all ----
{
  const ctx = stub()
  const r = 2
  // max reach is r1 + r2 * 2.5 = 2 + 5 = 7
  check("a reachable pair is joined", connector(ctx, 0, 0, r, 6.5, 0, r, GEO) === true)
  check("a pair beyond reach is not joined", connector(stub(), 0, 0, r, 7.5, 0, r, GEO) === false)
  check("a circle inside another is not joined", connector(stub(), 0, 0, 8, 1, 0, 2, GEO) === false)
  check("a zero radius is not joined", connector(stub(), 0, 0, 0, 5, 0, 3, GEO) === false)
}

// ---- 4. every subpath winds the same way ----
// this is the invariant the nonzero fill depends on. opposite windings would
// cancel where a membrane overlaps its circles, punching a hole through the join
// instead of filling it.
{
  const ctx = stub()
  paintInk(ctx, lattice(3, 3, Array(9).fill(1), Array(9).fill(0.9)), GEO, PALETTE, 0)

  // flatten the recorded path into subpaths, sampling beziers so the signed
  // area is meaningful, and check every one has the same sign
  const subpaths = []
  let current = null
  let cursor = null
  for (const o of ctx.ops) {
    if (o.op === "moveTo") {
      current = [{ x: o.x, y: o.y }]
      subpaths.push(current)
      cursor = { x: o.x, y: o.y }
    } else if (o.op === "lineTo" && current) {
      current.push({ x: o.x, y: o.y })
      cursor = { x: o.x, y: o.y }
    } else if (o.op === "bezier" && current) {
      for (let s = 1; s <= 12; s++) {
        const t = s / 12
        const u = 1 - t
        current.push({
          x:
            u * u * u * cursor.x +
            3 * u * u * t * o.c1x +
            3 * u * t * t * o.c2x +
            t * t * t * o.x,
          y:
            u * u * u * cursor.y +
            3 * u * u * t * o.c1y +
            3 * u * t * t * o.c2y +
            t * t * t * o.y,
        })
      }
      cursor = { x: o.x, y: o.y }
    } else if (o.op === "arc" && current) {
      // a full circle: sample it in the direction the arc was swept
      const steps = 24
      for (let s = 1; s <= steps; s++) {
        const a = o.a0 + ((o.a1 - o.a0) * s) / steps
        current.push({ x: o.x + Math.cos(a) * o.r, y: o.y + Math.sin(a) * o.r })
      }
      cursor = { x: o.x + Math.cos(o.a1) * o.r, y: o.y + Math.sin(o.a1) * o.r }
    }
  }

  const area = (pts) => {
    let sum = 0
    for (let i = 0; i < pts.length; i++) {
      const a = pts[i]
      const b = pts[(i + 1) % pts.length]
      sum += a.x * b.y - b.x * a.y
    }
    return sum / 2
  }
  const areas = subpaths.map(area).filter((a) => Math.abs(a) > 1e-9)
  const positive = areas.filter((a) => a > 0).length
  check("every subpath winds the same way", positive === areas.length || positive === 0,
    `${positive} of ${areas.length} positive`)
  check("the 3x3 block produced circles and membranes", subpaths.length > 9,
    `${subpaths.length} subpaths`)
}

// ---- 5. shades never fuse across colours ----
{
  const ctx = stub()
  paintInk(ctx, lattice(2, 1, [1, 2], [0.9, 0.9]), GEO, PALETTE, 0)
  check("different shades do not fuse", membranes(ctx.ops).length === 0,
    `${membranes(ctx.ops).length} membranes`)
}

// ---- 6. the dry shade draws its cells and never fuses ----
{
  const ctx = stub()
  paintInk(ctx, lattice(2, 1, [0, 0], [1, 1]), GEO, PALETTE, 0)
  check("dry shade draws its cells", ctx.ops.filter((o) => o.op === "arc").length === 2)
  check("dry shade draws no membranes", membranes(ctx.ops).length === 0)
  const arcs = ctx.ops.filter((o) => o.op === "arc")
  check("dry cells use the dry radius", arcs.every((a) => a.r === GEO.dryRadius),
    arcs.map((a) => a.r).join(","))
}

// ---- 7. diagonals fuse only when the L-route is dry ----
{
  const lone = stub()
  paintInk(lone, lattice(2, 2, [1, 0, 0, 1], [0.9, 0, 0, 0.9]), GEO, PALETTE, 0)
  check("a lone diagonal pair fuses diagonally", membranes(lone.ops).length === 2,
    `${membranes(lone.ops).length} bezier segments`)

  const solid = stub()
  paintInk(solid, lattice(2, 2, [1, 1, 1, 1], [0.9, 0.9, 0.9, 0.9]), GEO, PALETTE, 0)
  // four orthogonal pairs in a 2x2, two beziers each, and no diagonals
  check("a solid block fuses orthogonally only", membranes(solid.ops).length === 8,
    `${membranes(solid.ops).length} bezier segments`)
}

// ---- 8. ink grows with fill ----
{
  // fill 0 means absent, not faint: the faintest ink a cell can hold is just
  // above it
  const faint = stub()
  paintInk(faint, lattice(1, 1, [1], [0.001]), GEO, PALETTE, 0)
  const wet = stub()
  paintInk(wet, lattice(1, 1, [1], [1]), GEO, PALETTE, 0)
  const radius = (ops) => ops.find((o) => o.op === "arc").r
  check("a cell swells with its ink", radius(wet.ops) > radius(faint.ops),
    `${radius(faint.ops).toFixed(3)} -> ${radius(wet.ops).toFixed(3)}`)
  check("the faintest ink is the minimum radius", near(radius(faint.ops), GEO.minRadius, 0.01),
    `${radius(faint.ops).toFixed(3)} vs ${GEO.minRadius}`)
  check("a saturated cell is the maximum radius", near(radius(wet.ops), GEO.maxRadius, 1e-6),
    `${radius(wet.ops).toFixed(3)} vs ${GEO.maxRadius}`)
  check("an absent cell draws nothing",
    stub().ops.length === 0 &&
      (() => {
        const empty = stub()
        paintInk(empty, lattice(1, 1, [1], [0]), GEO, PALETTE, 0)
        return empty.ops.length === 0
      })())
  check("a saturated pair overlaps", GEO.maxRadius * 2 > PITCH,
    `${GEO.maxRadius * 2} vs pitch ${PITCH}`)
  check("the faintest ink cannot reach its neighbour",
    GEO.minRadius * (1 + GEO.reach) < PITCH,
    `${(GEO.minRadius * (1 + GEO.reach)).toFixed(2)} vs pitch ${PITCH}`)
}

// ---- 9. a saturated 2x2 block is plugged, a broken one is not ----
// four circles at the corners of a lattice square cover its edges but not its
// middle. left alone that pinhole reads as a perforation in what should be
// poured ink, so the square joining the four centres is filled - but only when
// all four corners are joined the whole way round.
{
  const solid = stub()
  paintInk(solid, lattice(2, 2, [1, 1, 1, 1], [0.9, 0.9, 0.9, 0.9]), GEO, PALETTE, 0)
  // the plug is the one subpath made only of lineTo, and its corners are the
  // four cell centres
  const plugs = []
  let run = null
  for (const o of solid.ops) {
    if (o.op === "moveTo") {
      run = { start: o, lines: [], curved: false }
    } else if (o.op === "lineTo" && run) {
      run.lines.push(o)
    } else if (o.op === "bezier" && run) {
      run.curved = true
    } else if (o.op === "close" && run) {
      if (!run.curved && run.lines.length === 3) plugs.push(run)
      run = null
    }
  }
  check("a saturated 2x2 block is plugged", plugs.length === 1, `${plugs.length} plugs`)
  if (plugs.length === 1) {
    const centres = [
      [5, 5],
      [15, 5],
      [15, 15],
      [5, 15],
    ]
    const got = [
      [plugs[0].start.x, plugs[0].start.y],
      ...plugs[0].lines.map((l) => [l.x, l.y]),
    ]
    const same = got.every((p, k) => near(p[0], centres[k][0]) && near(p[1], centres[k][1]))
    check("the plug's corners are the four cell centres", same, JSON.stringify(got))
  }

  // an open side means no plug. note which pairs actually fail to reach: the
  // minimum radius is deliberately large enough that a faint cell still reaches
  // a SATURATED neighbour, so a single faint corner does not open the square.
  // Two adjacent faint cells do - faint-to-faint is the pair that cannot span
  // the pitch - and then a plug would show outside the ink.
  check("a faint cell still reaches a saturated neighbour",
    GEO.minRadius + GEO.maxRadius * GEO.reach > PITCH,
    `${(GEO.minRadius + GEO.maxRadius * GEO.reach).toFixed(2)} vs pitch ${PITCH}`)

  const broken = stub()
  // row-major: [topLeft, topRight, bottomLeft, bottomRight] - the bottom pair is
  // faint, so the square's bottom edge holds no membrane
  paintInk(broken, lattice(2, 2, [1, 1, 1, 1], [0.9, 0.9, 0.02, 0.02]), GEO, PALETTE, 0)
  let brokenPlugs = 0
  let cur = null
  for (const o of broken.ops) {
    if (o.op === "moveTo") cur = { lines: 0, curved: false }
    else if (o.op === "lineTo" && cur) cur.lines++
    else if (o.op === "bezier" && cur) cur.curved = true
    else if (o.op === "close" && cur) {
      if (!cur.curved && cur.lines === 3) brokenPlugs++
      cur = null
    }
  }
  check("a block with an unreachable side is not plugged", brokenPlugs === 0,
    `${brokenPlugs} plugs`)
}

// ---- 10. one fill per inked shade ----
{
  const ctx = stub()
  paintInk(ctx, lattice(3, 1, [1, 2, 0], [0.9, 0.9, 0.9]), GEO, PALETTE, 0)
  const fills = ctx.ops.filter((o) => o.op === "fill")
  check("one fill per inked shade", fills.length === 3, fills.map((f) => f.style).join(","))
}

// ---- 11. swellIn grows a wetting cell in from nothing ----
// with swellIn set, the first stretch of a cell's fill is spent growing the
// dot from zero, so ink arrives as a swelling bead and leaves as a shrinking
// one instead of popping in and out at minRadius.
{
  const SWELL = { ...GEO, swellIn: 0.2 }
  const radius = (fill) => {
    const ctx = stub()
    paintInk(ctx, lattice(1, 1, [1], [fill]), SWELL, PALETTE, 0)
    const arc = ctx.ops.find((o) => o.op === "arc")
    return arc ? arc.r : 0
  }
  check("barely wet ink is barely there", radius(0.01) < GEO.minRadius * 0.1,
    `${radius(0.01).toFixed(4)} vs minRadius ${GEO.minRadius}`)
  check("the swell is monotonic", radius(0.05) < radius(0.1) && radius(0.1) < radius(0.2),
    `${radius(0.05).toFixed(3)} ${radius(0.1).toFixed(3)} ${radius(0.2).toFixed(3)}`)
  check("past the swell the radius is untouched",
    near(radius(0.2), GEO.minRadius + (GEO.maxRadius - GEO.minRadius) * 0.2, 1e-6) &&
      near(radius(1), GEO.maxRadius, 1e-6),
    `${radius(0.2).toFixed(3)} and ${radius(1).toFixed(3)}`)
  // and without swellIn nothing changes: pinned by section 8 above

  // a swelling cell is a bead, not yet a bond: a membrane attached to a
  // near-zero circle converges to a point and reads as a hair sticking out of
  // the mass. it bonds once its fill clears the swell.
  const swelling = stub()
  paintInk(swelling, lattice(2, 1, [1, 1], [0.05, 0.9]), SWELL, PALETTE, 0)
  check("a swelling cell holds no membrane yet", membranes(swelling.ops).length === 0,
    `${membranes(swelling.ops).length} bezier segments`)
  const bondedPair = stub()
  paintInk(bondedPair, lattice(2, 1, [1, 1], [0.3, 0.9]), SWELL, PALETTE, 0)
  check("a swollen cell bonds as before", membranes(bondedPair.ops).length === 2,
    `${membranes(bondedPair.ops).length} bezier segments`)
}

// Reversing unequal neighbours must preserve the liquid shape and its reach.
{
  const a = stub(), b = stub()
  const forward = connector(a, 0, 0, 2, 10, 0, 5, GEO)
  const reverse = connector(b, 10, 0, 5, 0, 0, 2, GEO)
  check("unequal neighbours join in either direction", forward && reverse)
  check("unequal neighbours release in either direction",
    !connector(stub(), 0, 0, 2, 12.3, 0, 5, GEO) &&
    !connector(stub(), 12.3, 0, 5, 0, 0, 2, GEO))
  const nearRelease = stub()
  connector(nearRelease, 0, 0, 4, 13.999, 0, 4, GEO)
  const points = nearRelease.ops.filter(o => o.op === "moveTo" || o.op === "bezier")
  check("the neck closes before release", points.every(p => Math.abs(p.y) < 0.00001))
  check("a closing neck never folds across itself", membranes(nearRelease.ops).every(p =>
    p.y < 0 ? p.c1y <= 0 && p.c2y <= 0 : p.c1y >= 0 && p.c2y >= 0))
  check("reach zero also disables overlapping connectors",
    !connector(stub(), 0, 0, 5, 6, 0, 5, { ...GEO, reach: 0 }))
}

// Curve work should grow with a solid region's perimeter, not its area.
{
  const ctx = stub()
  const cells = lattice(20, 20, Array(400).fill(1), Array(400).fill(1))
  paintInk(ctx, cells, GEO, PALETTE, 0)
  const arcs = ctx.ops.filter(o => o.op === "arc").length
  check("solid interiors omit invisible circles", arcs === 76, `${arcs} circles for 400 cells`)
  check("solid interiors omit invisible membranes", membranes(ctx.ops).length <= 152,
    `${membranes(ctx.ops).length} curves for 400 cells`)
  cells.fill[210] = 0
  const hole = stub()
  paintInk(hole, cells, GEO, PALETTE, 0)
  check("opening a hole restores its boundary geometry",
    hole.ops.filter(o => o.op === "arc").length > arcs)
  const thin = stub()
  paintInk(thin, lattice(2, 2, [1, 1, 1, 1], [0.6, 0.6, 0.6, 0.6]), GEO, PALETTE, 0)
  check("thin membranes do not flood a square's centre",
    thin.ops.filter(o => o.op === "lineTo").length === 4)
}

console.log(failures === 0 ? "\nall ink geometry checks passed" : `\n${failures} FAILED`)
process.exit(failures === 0 ? 0 : 1)
