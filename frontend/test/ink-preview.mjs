// Visual smoke check: run the real renderer against a context that records an
// SVG path instead of rasterising, so the output can be looked at.
import { readFileSync, writeFileSync } from "node:fs"
import { createRequire } from "node:module"
import { fileURLToPath } from "node:url"
import path from "node:path"
import vm from "node:vm"

const require = createRequire(import.meta.url)
const ts = require("typescript")
const here = path.dirname(fileURLToPath(import.meta.url))
const source = readFileSync(path.join(here, "..", "lib", "ink-render.ts"), "utf8")
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText
const mod = { exports: {} }
vm.runInNewContext(compiled, { module: mod, exports: mod.exports, Math, console })
const { paintInk, gridNeighbours } = mod.exports

// an SVG-emitting 2D context
function svgCtx() {
  let d = ""
  const paths = []
  let cur = { x: 0, y: 0 }
  const n = (v) => Math.round(v * 100) / 100
  return {
    fillStyle: "",
    beginPath() {
      d = ""
    },
    moveTo(x, y) {
      d += `M${n(x)} ${n(y)}`
      cur = { x, y }
    },
    lineTo(x, y) {
      d += `L${n(x)} ${n(y)}`
      cur = { x, y }
    },
    arc(x, y, r, a0, a1) {
      // a full circle as two arcs
      d += `M${n(x + r)} ${n(y)}A${n(r)} ${n(r)} 0 1 1 ${n(x - r)} ${n(y)}A${n(r)} ${n(r)} 0 1 1 ${n(x + r)} ${n(y)}`
    },
    bezierCurveTo(c1x, c1y, c2x, c2y, x, y) {
      d += `C${n(c1x)} ${n(c1y)} ${n(c2x)} ${n(c2y)} ${n(x)} ${n(y)}`
      cur = { x, y }
    },
    closePath() {
      d += "Z"
    },
    fill() {
      paths.push({ d, fill: this.fillStyle })
    },
    paths,
  }
}

const PITCH = 40
const GEO = {
  minRadius: 0.22 * PITCH,
  maxRadius: 0.54 * PITCH,
  spread: 0.5,
  handleSize: 2.4,
  reach: 2.5,
  dryRadius: 0.2 * PITCH,
  diagonals: true,
}

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

// scenes chosen to show the thing the reference image shows: a joined pair at
// several separations, a diagonal run, and a growing front
const scenes = [
  { name: "pair, full ink", w: 2, h: 1, shade: [1, 1], fill: [1, 1] },
  { name: "pair, mid ink", w: 2, h: 1, shade: [1, 1], fill: [0.62, 0.62] },
  { name: "pair, uneven", w: 2, h: 1, shade: [1, 1], fill: [1, 0.45] },
  { name: "row of four", w: 4, h: 1, shade: [1, 1, 1, 1], fill: [0.5, 0.85, 0.7, 0.4] },
  { name: "diagonal run", w: 3, h: 3, shade: [1, 0, 0, 0, 1, 0, 0, 0, 1], fill: [0.8, 0, 0, 0, 0.8, 0, 0, 0, 0.8] },
  {
    name: "growing front",
    w: 5,
    h: 3,
    shade: Array(15).fill(1),
    fill: [
      0.15, 0.4, 0.7, 0.95, 1,
      0.3, 0.6, 0.9, 1, 1,
      0.1, 0.3, 0.55, 0.8, 0.95,
    ],
  },
]

let body = ""
let oy = 0
const COLW = 5 * PITCH + 40
let ox = 0
let rowH = 0
for (const s of scenes) {
  const ctx = svgCtx()
  paintInk(ctx, lattice(s.w, s.h, s.shade, s.fill), GEO, ["#dry", "#141414"], 0)
  const w = s.w * PITCH
  const h = s.h * PITCH
  if (ox + w + 40 > COLW * 2 + 80) {
    ox = 0
    oy += rowH + 50
    rowH = 0
  }
  body += `<g transform="translate(${ox + 20} ${oy + 30})">`
  body += `<text x="0" y="-10" font-family="monospace" font-size="12" fill="#888">${s.name}</text>`
  for (const p of ctx.paths) {
    body += `<path d="${p.d}" fill="${p.fill}" fill-rule="nonzero"/>`
  }
  body += `</g>`
  ox += w + 60
  rowH = Math.max(rowH, h)
}

const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${COLW * 2 + 100}" height="${oy + rowH + 70}" viewBox="0 0 ${COLW * 2 + 100} ${oy + rowH + 70}"><rect width="100%" height="100%" fill="#fff"/>${body}</svg>`
const out = path.join(here, "..", "..", "ink-preview.svg")
writeFileSync(out, svg)
console.log(`wrote ${out}`)
